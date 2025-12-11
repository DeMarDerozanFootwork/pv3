/*******************************************************************************
 * @file    pv3.cpp
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#include "pv3.h"

#include "driver.h"
#include "tidl_kernels.h"
#include "utils.h"

namespace phantom_ai {
namespace driver {
namespace pv3 {
namespace {

// Decode
struct Decode final {
  // PV3 compile-time inference decoding configuration parameters
  using Configuration = vision::core::Configuration::Inference::Main::Decode;

  // PV3 initialization-time inference decoding configuration parameters
  using Parameters = vision::inference::decode::Parameters::Main;

  // PV3 runtime inference decoding output
  using Output = vision::Output::Inference::Main::Decode;
};

/*
  Callbacks
*/

//
// Note: Do not perform any computationally intensive operations on any of the
// below PV3 callbacks!  These callbacks must be kept as light as possible with
// any extra processing scheduled for a later time on a worker thread. Think of
// Linux ISR top-halves and bottom-halves with these callbacks as the former.
// To elaborate further, keep in mind that PV3 is the front end of the vision
// pipeline, and produces data that the rest of the perception stack has a
// blocking dependency on.  Combined with the fact that these callbacks run
// inline on the primary PV3 thread that is in charge of orchestrating all PV3
// jobs, any slowdowns here will impact the entirety of the vision pipeline!
//

namespace callbacks {

void
on_capture(                   //
    driver::Context& driver,  //
    const vision::Output::Camera::Capture& /*capture*/) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // Note 1: This code is not necessary if PV3 is run in internal capture mode.
  // Note 2: We leave vehicle state data intact.

  /* Critical Section */ {
    // All access to the queue must be synchronized!
    pv3::Context::Queue::Input::Image& queue = driver.pv3.queue.input.image;

    // Clear the queue.
    queue.nv12.clear();
    queue.storage.clear();

    // Refer to the comment titled 'Input Queue Synchronization' in process().
    queue.mutex.unlock();
  } /* Critical Section */
}

void
on_convert(                   //
    driver::Context& driver,  //
    const vision::Output::Image::Convert& convert) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // Don't bother if either grayscale or RGB color conversions failed for
  // whatever reason.  Technically, at the expense of additional code
  // complexity, we can either fully or partially reconstruct one from the other
  // if one succeeds and the other doesn't, but this is not likely given the
  // current implementation details of PV3.  Still, we could add support for
  // that behavior if we ever need to.
  if ((Status::Success != convert.gray.status) ||
      (Status::Success != convert.rgb.status)) {
    return;
  }

  // All access to the queue must be synchronized!
  pv3::Context::Queue::Output::Convert& queue = driver.pv3.queue.output.convert;

  /* Critical Section */ {
    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // We have the following two options in dealing with this queue: (1) either
    // block further production until all previously produced data is consumed,
    // or (2) drop data not yet consumed and replace it with a more recent
    // batch.  This implementation chooses the 2nd approach.  Note that there
    // will be  perceptible difference between the two options if consumption
    // keeps up with production at all times.  Otherwise, the first approach may
    // intorduce some jitter but will have the advantage of the system gathering
    // and processing more information from the environment.

    // Clear the queue in case it is not yet consumed which would happen if we
    // keep producing output faster than the background thread can process.
    queue.gray.clear();
    queue.rgb.clear();

    // Get the most recently converted grayscale images.  This operation simply
    // copies the image handles and does not move any substansive amount of
    // data. Remember that our goal with PV3 callbacks is to return ASAP.
    std::copy(                                 //
        convert.gray.images.front().cbegin(),  //
        convert.gray.images.front().cend(),    //
        std::back_inserter(queue.gray));

    // Get the most recently converted RGB images.  This operation simply copies
    // the image handles and does not move any substansive amount of data.
    // Remember that our goal with PV3 callbacks is to return ASAP.
    std::copy(                                //
        convert.rgb.images.front().cbegin(),  //
        convert.rgb.images.front().cend(),    //
        std::back_inserter(queue.rgb));

    // Track the frame numbers this data corresponds to.
    queue.frame = convert.rgb.frames.front();
  } /* Critical Section */

  // Notify the worker thread that a new output is ready.
  queue.produced.notify_one();

  // Note: Do not yield.  This function is on the critical path.
}

void
on_inference_main(            //
    driver::Context& driver,  //
    const vision::Output::Inference::Main& inference) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // We are only interested in decoding results for now.
  const pv3::Decode::Output& decode = inference.decode;

  // Don't bother if inference decoding failed for whatever reason.
  if (Status::Success != decode.status) {
    return;
  }

  // All access to the queue must be synchronized!
  pv3::Context::Queue::Output::Decode& queue = driver.pv3.queue.output.decode;

  /* Critical Section */ {
    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // We have the following two options in dealing with this queue: (1) either
    // block further production until all previously produced data is consumed,
    // or (2) drop data not yet consumed and replace it with a more recent
    // batch.  This implementation chooses the 2nd approach.  Note that there
    // will be  perceptible difference between the two options if consumption
    // keeps up with production at all times.  Otherwise, the first approach may
    // introduce some jitter but will have the advantage of the system gathering
    // and processing more information from the environment.

    // Override the queue contents even if its contents are yet to be consumed
    // by the worker thread which would happen if we keep producing output
    // faster than the background thread can process.  This operation does not
    // copy any substantive amount of data.  Remember that our goal with PV3
    // callbacks is to return as soon as possible.
    queue.data = decode;

    // Track the frame numbers this data corresponds to.
    queue.frame = decode.frames.front();
  } /* Critical Section */

  // Notify the worker thread that a new output is ready.
  queue.produced.notify_one();

  // Note: Do not yield.  This function is on the critical path.
}

}  // namespace callbacks

/*
  Workers
*/

//
// Note: It is safer to perform computationally intenstive operations on these
// background worker threads.  Still, keep in mind that there is no free lunch
// in computing in that these threads will still be competing with the rest of
// the system for CPU time.  Try to keep things light!  Following our Linux ISR
// top-halves and bottom-halves analogy, these workers will be the latter.
//

namespace workers {

void
convert(driver::Context& driver) {
  // Alias
  pv2::Context& pv2 = driver.pv2;
  pv3::Context& pv3 = driver.pv3;

  while (true) {
    CamerasDataS camera{};
    VehicleStateDataS vsd{};

    /* Critical Section */ {
      // All access to the queue must be synchronized!
      pv3::Context::Queue::Output::Convert& queue = pv3.queue.output.convert;

      // We cannot modify the queue if its contents are modified in parallel.
      std::unique_lock<std::mutex> lock(queue.mutex);

      // Wait until we either have work to do or are instructed to terminate.
      queue.produced.wait(lock, [&queue]() noexcept {
        return !queue.rgb.empty() || queue.quit;
      });

      // Are we woken up with a termination request?
      if (queue.quit) {
        break;
      }

      // Create a new PV2 payload for this data.
      camera = std::make_shared<CamerasData>(T_NOW, NUM_MAX_CAM_IDS);

      // Copy the images over.  We could technically avoid these memory copies
      // and instead use the internal PV3 memory pointed to by these handles
      // directly if we could somehow guarantee that the created CamerasData
      // object would have been completely consumed and destroyed within N
      // frames, where N is the broadcasted lifetime of the data pointed to by
      // these handles, but since we cannot easily guarantee that with certainty
      // given PV2's reliance on shared ownership semantics (i.e. the object may
      // stay alive for an indeterminate period of time after it's been created
      // and sent out as long as someone, somewhere, holds a reference to it,
      // even accidentally), we must perform a deep copy to be on the safe side.

      for (uint32_t view = 0U; view < queue.rgb.size(); ++view) {
        // Get the PV3 image handle corresponding to this converted RGB image.
        const struct {
          vision::image::Handle gray, rgb;
        } src{
            /* .gray = */ queue.gray[view],
            /* .rgb = */ queue.rgb[view],
        };

        // Create new OpenCV matricies.  This will dynamically allocate memory.
        struct {
          cv::Mat gray, rgb;
        } dst{
            /* .gray = */ cv::Mat(src.gray.height, src.gray.width, CV_8UC1),
            /* .rgb = */ cv::Mat(src.rgb.height, src.rgb.width, CV_8UC3),
        };

        // Copy the converted grayscale image over.
        std::memcpy(                    //
            dst.gray.ptr(),             //
            src.gray.data.as_void_ptr,  //
            dst.gray.total() * dst.gray.elemSize());

        // Copy the converted rgb image over.
        std::memcpy(                   //
            dst.rgb.ptr(),             //
            src.rgb.data.as_void_ptr,  //
            dst.rgb.total() * dst.rgb.elemSize());

        // Add the image to its PV2 container.
        camera->Insert(            //
            pv2.camera.map[view],  //
            T_NOW_SEC,             //
            false,                 //
            dst.rgb,               //
            dst.gray);
      }

      // Set metadata.
      camera->frame() = queue.frame;
      camera->t_hw() = T_NOW;

      // Done!
      queue.rgb.clear();
    } /* Critical Section */

    // Post & Draw
    if (camera) {
      // Draw only if visualization is enabled.  Assuming params is read-only.
      if (pv2.params.visualizer_task.enable_task) {
        VisualizerDataS canvas = std::make_shared<VisualizerData>(camera);
        canvas->Initialize(pv2.params.visualizer_window);
        canvas->Draw(camera, VIZ_DRAW_CAMERAS_COLOR);
        pv2.messages[misc::to_underlying(VisionMessageID::VISUALIZER)]
            .AddSharedData(canvas);
      }

      /* Critical Section */ {
        // All access to the queue must be synchronized!
        pv3::Context::Queue::Input::Vehicle& queue = pv3.queue.input.vehicle;

        // We cannot modify the queue if its contents are modified in parallel.
        std::unique_lock<std::mutex> lock(queue.mutex);

        // Create a new PV2 payload for this data.
        vsd = std::make_shared<VehicleStateData>(         //
            T_NOW,                                        //
            queue.esp,                                    //
            queue.wsm,                                    //
            queue.sa,                                     //
            queue.gs,                                     //
            pv2.params.vehicle_setting.drive_train_type,  //
            queue.yaw.rate);

        // Set metadata.
        vsd->frame() = camera->frame();

        // Post the vehicle state to all interested parties. Note that PV2's
        // message queue is thread-safe.  No extra locking is required.
        pv2.messages[misc::to_underlying(VisionMessageID::VEHICLE_STATE_ROS)]
            .AddSharedData(vsd);
      } /* Critical Section */

      // Post the captured images to all interested parties. Note that PV2's
      // message queue is thread-safe.  No extra locking mechanism is required.
      pv2.messages[misc::to_underlying(VisionMessageID::CAMERAS)]  //
          .AddSharedData(camera);
    }
  }
}

void
inference(driver::Context& driver) {
  pv2::Context& pv2 = driver.pv2;
  pv3::Context& pv3 = driver.pv3;

  // An array of Phantom Net data objects, one per view. This definition is
  // pulled out of the inner loop to minimize unnecessary memory allocations as
  // vector::clear() leaves vector::capacity() unchanged.
  std::vector<PhantomnetDataS> payloads;
  payloads.reserve(pv3.camera.views.size());

  // An array of object detection candidates. This definition is pulled out of
  // the inner loop to minimize unnecessary memory allocations as
  // vector::clear() leaves vector::capacity() unchanged.
  std::vector<phantom_ai::DetectionBoxS> candidates;

  // Configuration
  constexpr bool kRemap = true;
  constexpr bool kNMS = false;  // Handled in PV3!

  while (true) {
    /* Critical Section */ {
      // All access to the queue must be synchronized!
      pv3::Context::Queue::Output::Decode& queue = pv3.queue.output.decode;

      // We cannot modify the queue if its contents are modified in parallel.
      std::unique_lock<std::mutex> lock(queue.mutex);

      // Wait until we either have work to do or are instructed to terminate.
      queue.produced.wait(lock, [&queue]() noexcept {
        return (Status::Success == queue.data.status) || queue.quit;
      });

      // Are we woken up with a termination request?
      if (queue.quit) {
        break;
      }

      struct {
        MessageQueue& queue;
        CamerasDataS payload;
      } camera{
          /* .queue = */
          pv2.messages[misc::to_underlying(VisionMessageID::CAMERAS)],

          /* .payload = */
          camera.queue.PeekMessageFrameAt<CamerasData>(  //
              TASK_FEATURE_TRACK,                        //
              queue.frame),
      };

      if (camera.payload) {
        // Reusing the vector to avoid unnecessary memory allocations.
        payloads.clear();

        // Global detection id
        int32_t id = 0;

        for (uint32_t view = 0U; view < pv3.camera.views.size(); ++view) {
          // Create a new PV2 payload for this data.
          const PhantomnetDataS payload = std::make_shared<PhantomnetData>(  //
              camera.payload,                                                //
              pv2.camera.map[view]);

          /*
            Center
          */

          {}

          /*
            Horizon
          */

          {
            // Source (PV3)
            using Horizon = vision::inference::decode::Horizon;
            const Horizon* const src = queue.data.horizons.front()[view];
            constexpr size_t kLine = pv3::Decode::Configuration::Horizon::kLine;

            // Destination (PV2)
            HorizonLine& dst = payload->horizon();
            dst.valid = true;
            dst.horizon = src->horizon;
            dst.probability = src->score;
            dst.best_index = src->horizon;

            // Copy!
            dst.probabilities.resize(kLine);
            std::copy(src->scores, src->scores + kLine,
                      dst.probabilities.begin());
          }

          /*
            Object
          */

          {
            // Source (PV3)
            using Object = vision::inference::decode::Object;
            const Object* const object = queue.data.objects.front()[view];

            // PV2 detections really shouldn't have been an array of shared
            // pointers.  The number of per-frame allocations and cache misses
            // incurred in total is truly excessive.

            // Destination (PV2)
            using Detections = std::vector<phantom_ai::DetectionBoxS>;
            Detections& dst = payload->detections();

            // Total number of detections
            uint32_t detections = 0U;

            // For each box decoder
            for (uint32_t idecoder = 0U;          //
                 idecoder < object->boxes.count;  //
                 ++idecoder) {
              // Tally up the total number of detections.
              detections += object->boxes.array[idecoder].count;
            }

            // For each cuboid decoder
            for (uint32_t idecoder = 0U;            //
                 idecoder < object->cuboids.count;  //
                 ++idecoder) {
              // Tally up the total number of detections.
              detections += object->cuboids.array[idecoder].count;
            }

            // For each VRU decoder
            for (uint32_t idecoder = 0U;         //
                 idecoder < object->vrus.count;  //
                 ++idecoder) {
              // Tally up the total number of detections.
              detections += object->vrus.array[idecoder].count;
            }

            // Aim to minimize the total number of allocations in this loop by
            // allocating enough memory up front for the worst case scenario.
            // We're doing way too many allocations in this function as is.
            dst.reserve(detections);

            /*
              Boxes
            */

            {
              // Source (PV3)
              using Decoders = Object::Decoder::Boxes;
              const Decoders& src = object->boxes;

              // Masks (PV2)
              using Mask = std::vector<cv::Point>;
              using Masks = std::vector<Mask>;

              // Assuming HAL has published its output in the order configured.
              const Masks& masks = pv2.params.cameras_image_mask[view].regions;

              // Class Mappings
              const struct {
                const std::vector<ClassificationList>& main;
                const std::vector<ClassificationList>& traffic;
              } classes{
                  /*. main = */ pv3.network.front.classes["bbox"],
                  /*. traffic = */ pv3.network.front.classes["traffic"],
              };

              // For each box decoder
              for (uint32_t idecoder = 0U; idecoder < src.count; ++idecoder) {
                using Boxes = Object::Detection::Boxes;
                const Boxes& decoder = src.array[idecoder];

                // Reusing the vector to avoid unnecessary memory allocations.
                candidates.clear();

                // For each detection
                for (uint32_t idetection = 0U;    //
                     idetection < decoder.count;  //
                     ++idetection) {
                  using Box = Object::Box;
                  const Box& box = decoder.array[idetection];
                  const vision::core::geometry::Rectf32& bounds = box.bounds;

                  // Is the bounding box masked?
                  const bool masked = std::any_of(
                      masks.cbegin(), masks.cend(),
                      [&bounds](const Mask& mask) {
                        const cv::Point2f point{
                            bounds.origin.x() + (0.5F * bounds.size.x()),
                            bounds.origin.y() + (0.5F * bounds.size.y()),
                        };

                        return cv::pointPolygonTest(mask, point, false) > 0.0;
                      });

                  if (!masked) {
                    // Convert PV3 detection to its PV2 equivalent.
                    candidates.push_back(std::make_shared<DetectionBox>(  //
                        pv2.camera.map[view],                             //
                        id++,                                             //
                        box.classification,                               //
                        box.score,                                        //
                        cv::Rect2f{
                            bounds.origin[0U],
                            bounds.origin[1U],
                            bounds.size[0U],
                            bounds.size[1U],
                        },
                        -1));
                  }
                }

                // If classification id remapping is successful,
                if (remap_classification_phantomnet_boxes(  //
                        candidates,                         //
                        kRemap,                             //
                        classes.main[idecoder],             //
                        classes.traffic)) {
                  // Sort and add to the final list.
                  suppress_sort_phantomnet_boxes(  //
                      candidates,                  //
                      pv2.camera.map[view],        //
                      kNMS,                        //
                      0.3F,                        //
                      0.3F,                        //
                      dst);
                }
              }
            }

            /*
              Cuboids
            */

            {
              // Source (PV3)
              using Decoders = Object::Decoder::Cuboids;
              const Decoders& src = object->cuboids;

              // For each cuboid decoder
              for (uint32_t idecoder = 0U; idecoder < src.count; ++idecoder) {
                using Cuboids = Object::Detection::Cuboids;
                const Cuboids& decoder = src.array[idecoder];

                // Reusing the vector to avoid unnecessary memory allocations.
                candidates.clear();

                // For each detection
                for (uint32_t idetection = 0U;    //
                     idetection < decoder.count;  //
                     ++idetection) {
                  using Cuboid = Object::Cuboid;
                  const Cuboid& cuboid = decoder.array[idetection];

                  // Convert PV3 detection to its PV2 equivalent.
                  candidates.push_back(std::make_shared<DetectionBox>(  //
                      pv2.camera.map[view],                             //
                      id++,                                             //
                      BBOX_CUBOID_FACE_UNKNOWN +                        //
                          cuboid.classification.face +                  //
                          (3U * cuboid.classification.side),            //
                      cuboid.score.face,                                //
                      cv::Rect2f{
                          cuboid.box.origin[0U],
                          cuboid.box.origin[1U],
                          cuboid.box.size[0U],
                          cuboid.box.size[1U],
                      },
                      -1));

                  candidates.back()->extra_info.push_back({
                      cuboid.score.face,
                      cv::Point2f{
                          cuboid.side.left[0U],
                          cuboid.side.left[1U],
                      },
                      cv::Point2f{
                          cuboid.side.right[0U],
                          cuboid.side.right[1U],
                      },
                  });
                }

                // Sort and add to the final list.
                suppress_sort_phantomnet_boxes(  //
                    candidates,                  //
                    pv2.camera.map[view],        //
                    kNMS,                        //
                    0.3F,                        //
                    0.3F,                        //
                    dst);
              }
            }

            /*
              VRUs
            */

            {
              // Source (PV3)
              using Decoders = Object::Decoder::VRUs;
              const Decoders& src = object->vrus;

              // Class Mappings
              const struct {
                const std::vector<ClassificationList>& main;
              } classes{
                  /*. main = */ pv3.network.front.classes["vru"],
              };

              // For each VRU decoder
              for (uint32_t idecoder = 0U; idecoder < src.count; ++idecoder) {
                using VRUs = Object::Detection::VRUs;
                const VRUs& decoder = src.array[idecoder];

                // Reusing the vector to avoid unnecessary memory allocations.
                candidates.clear();

                // For each detection
                for (uint32_t idetection = 0U;    //
                     idetection < decoder.count;  //
                     ++idetection) {
                  using VRU = Object::VRU;
                  const VRU& vru = decoder.array[idetection];

                  // Convert PV3 detection to its PV2 equivalent.
                  candidates.push_back(std::make_shared<DetectionBox>(  //
                      pv2.camera.map[view],                             //
                      id++,                                             //
                      vru.object.classification,                        //
                      vru.object.score,                                 //
                      cv::Rect2f{
                          vru.object.bounds.origin[0U],
                          vru.object.bounds.origin[1U],
                          vru.object.bounds.size[0U],
                          vru.object.bounds.size[1U],
                      },
                      -1));

                  candidates.back()->extra_info.push_back({
                      vru.torso.score,
                      cv::Rect2f{
                          vru.torso.bounds.origin[0U],
                          vru.torso.bounds.origin[1U],
                          vru.torso.bounds.size[0U],
                          vru.torso.bounds.size[1U],
                      },
                  });
                }

                // If classification id remapping is successful,
                if (remap_classification_phantomnet_boxes(  //
                        candidates,                         //
                        kRemap,                             //
                        classes.main[idecoder])) {
                  // Sort and add to the final list.
                  suppress_sort_phantomnet_boxes(  //
                      candidates,                  //
                      pv2.camera.map[view],        //
                      kNMS,                        //
                      0.3F,                        //
                      0.3F,                        //
                      dst);
                }
              }
            }
          }

          /*
            Segmentation
          */

          {
            // Source (PV3)
            using Masks = pv3::Decode::Output::Spatial::Masks;
            const Masks src = queue.data.segmentation.front()[view];

            // We assume the mask has the same dimensions as the original image.
            const cv::Size size = payload->image_size();

            // Object Segmentation
            if (!src.empty() && src[0U]) {
              // Allocate memory for the object segmentation mask.
              cv::Mat object(size, CV_8UC1);

              // Decode the segmentation mask.
              utils::decode(object, src[0U], {pv3.network.front.map});

              // Pass on the matrices.
              payload->seg() = object;

              // Generate and blend the segmentation masks overlay only if
              // visualization is enabled.  Assume params is read-only.
              if (pv2.params.visualizer_task.enable_task &&
                  (0 == pv3.network.visualizer.id)) {
                payload->viz() = cv::Mat(size, CV_8UC3);

                utils::visualize(                                 //
                    payload->viz(),                               //
                    camera.payload->image(pv2.camera.map[view]),  //
                    payload->seg(),                               //
                    {pv3.network.visualizer.b},                   //
                    {pv3.network.visualizer.g},                   //
                    {pv3.network.visualizer.r},                   //
                    {pv3.network.visualizer.a});
              }
            }

            // Lane Segmentation
            if ((src.count() > 1U) && src[1U]) {
              // Allocate memory for the lane segmentation mask.
              cv::Mat lane(size, CV_8UC1);

              // Decode the segmentation mask.
              utils::decode(lane, src[1U], {pv3.network.traffic.map});

              // Pass on the matrices.
              payload->lane_seg() = lane;

              // Generate and blend the segmentation masks overlay only if
              // visualization is enabled.  Assume params is read-only.
              if (pv2.params.visualizer_task.enable_task &&
                  (1 == pv3.network.visualizer.id)) {
                payload->viz() = cv::Mat(size, CV_8UC3);

                utils::visualize(                                 //
                    payload->viz(),                               //
                    camera.payload->image(pv2.camera.map[view]),  //
                    payload->lane_seg(),                          //
                    {pv3.network.visualizer.b},                   //
                    {pv3.network.visualizer.g},                   //
                    {pv3.network.visualizer.r},                   //
                    {pv3.network.visualizer.a});
              }
            }
          }

          // Generate objects.
          generate_phantomnet_objects(  //
              payload->detections(),    //
              payload->seg(),           //
              payload->get_objects_ref_without_lock());

          // Add the payload to the list.
          payloads.push_back(payload);
        }
      }

      // Done!
      queue.data.status = Status::Failure;
    } /* Critical Section */

    // Post & Draw
    for (PhantomnetDataS payload : payloads) {
      // Draw only if visualization is enabled.  Assuming params is read-only.
      if (pv2.params.visualizer_task.enable_task) {
        struct {
          MessageQueue& queue;
          const VisualizerDataS payload;
        } visualizer{
            /* .queue = */
            pv2.messages[misc::to_underlying(VisionMessageID::VISUALIZER)],

            /* .payload = */
            visualizer.queue.PeekMessageFrameAt<VisualizerData>(  //
                TASK_VISUALIZER,                                  //
                payload->frame()),
        };

        // If found a viz message for this frame,
        if (visualizer.payload) {
          // Draw!
          visualizer.payload->Draw(  //
              payload,               //
              pv2.params.visualizer_task.drawing_mode_phantomnet);
        }
      }

      // Post the captured images to all interested parties. Note that PV2's
      // message queue is thread-safe.  No extra locking mechanism is required.
      pv2.messages[misc::to_underlying(VisionMessageID::PHANTOMNET_A)]
          .AddSharedData(payload);
    }
  }
}

}  // namespace workers

/*
  Utilities
*/

Decode::Parameters::Object::Decoder::Type
convert(TidlDecoderType type) noexcept {
  using Object = Decode::Parameters::Object;
  Object::Decoder::Type dtype{Object::Decoder::Type::Undefined};

  switch (type) {
    case TidlDecoderType::BBOX:
      dtype = Object::Decoder::Type::Box;
      break;

    case TidlDecoderType::CUBOID:
      dtype = Object::Decoder::Type::Cuboid;
      break;

    case TidlDecoderType::VRUBOX:
      dtype = Object::Decoder::Type::VRU_AABB;
      break;

    case TidlDecoderType::VRUBOX_W_ORIENTATION:
      dtype = Object::Decoder::Type::VRU_OBB;
      break;

    case TidlDecoderType::CLASSIFIER:
    case TidlDecoderType::MULTI_CLASSIFIER:
    case TidlDecoderType::UNKNOWN:
    default:
      break;
  }

  PV3_ASSERT(Object::Decoder::Type::Undefined != dtype, "Invalid!");
  return dtype;
}

}  // namespace

Status
initialize(driver::Context& driver) {
  // Alias
  pv3::Context& pv3 = driver.pv3;

  // Pay special attention that all initialization-time objects whose lifetime
  // must exceed that of vision::Parameters must be stored below!  Failure to do
  // so will result in undefined behavior as a result of vision::initialize()
  // accessing reclaimed stack memory locations that no longer belong to the
  // intended objects!  To elaborate, this is pretty straightforward to
  // anticipate with naked pointers: if you pass to a function, a pointer to an
  // object, and have the object go out of scope before the function returns,
  // then you would expect to access invalid memory, right?  Where this might be
  // a little less obvious would be with nested objects.  For instance, if you
  // are passing an object to a function which itself is holding a pointer to
  // another object that may go out of scope before the function returns.  All
  // thin objects that do not own their underlying memory, such as std::span or
  // vision::Span fall into this latter category and must be treated with care!

  struct {
    struct {
      struct {
        std::vector<vision::camera::Sensor> sensors;
        std::vector<vision::camera::Crop> crops;
      } capture;
    } camera;

    struct {
      struct {
        TidlModelConfigs configs;

        struct {
          TidlParams front;
          TidlParams traffic;
        } models;
      } tidl;

      struct {
        struct {
          std::vector<vision::core::geometry::Point2f32> ratios;
          std::vector<Decode::Parameters::Object::Decoder> decoders;
          uint32_t tsr;
        } object;
      } decode;
    } inference;

    // PV3 Initialization Parameters
    vision::Parameters init;
  } params{};

  //
  // Camera
  //

  {
    // For each detected camera
    for (uint16_t channel = 0U;                          //
         channel < driver.capture.params.camera.size();  //
         ++channel) {
      // Get the parameters for the camera at this location.
      const auto& camera = driver.capture.params.camera[channel];

      // Ensure the output format is supported!
      if (vision::core::image::Format::Unknown ==
          utils::convert(camera.output_format)) {
        return Status::Failure;
      }

      // Store sensor configurations.
      params.camera.capture.sensors.push_back({
          /* .format = */ utils::convert(camera.output_format),
          /* .width = */ cast::numeric<uint16_t>(camera.sensor_width),
          /* .height = */ cast::numeric<uint16_t>(camera.sensor_height),
      });

      // For each requested subimage from that camera
      for (const auto& subimage : camera.subimage) {
        // Convert PV2 CameraIDs to their corresponding PV3 camera::Views.
        const vision::camera::View view = utils::convert(subimage.location);

        // Store the view id for runtime use.
        pv3.camera.map[subimage.location] = pv3.camera.views.size();
        pv3.camera.views.push_back(view);

        // Convert HAL / TDA4x subimages to PV3 crops.
        params.camera.capture.crops.push_back({
            /* .input = */
            {
                /* .channel = */ channel,
                /* .roi = */
                {
                    /* .origin = */
                    {
                        cast::numeric<uint16_t>(subimage.roi.x),
                        cast::numeric<uint16_t>(subimage.roi.y),
                    },
                    /*. size = */
                    {
                        cast::numeric<uint16_t>(subimage.roi.width),
                        cast::numeric<uint16_t>(subimage.roi.height),
                    },
                },
            },
            /* .output = */
            {
                /* .view = */ view,
                /* .size = */
                {
                    cast::numeric<uint16_t>(subimage.output_width),
                    cast::numeric<uint16_t>(subimage.output_height),
                },
            },
        });
      }
    }

    /*
      PV3
    */

    params.init.camera.capture = {
        /* .frequency = */ 15U,
        /* .optimize = */ vision::Parameters::Optimize::Latency,
        /* .sensors = */
        {
            /* .data = */
            params.camera.capture.sensors.data(),

            /* .count = */
            cast::numeric<uint32_t>(params.camera.capture.sensors.size()),
        },
        /* .crops = */
        {
            /* .data = */
            params.camera.capture.crops.data(),

            /* .count = */
            cast::numeric<uint32_t>(params.camera.capture.crops.size()),
        },
        /* .callback = */
        std::bind(                  //
            callbacks::on_capture,  //
            std::ref(driver),       //
            std::placeholders::_1),
    };
  }

  //
  // Inference
  //

  {
    auto& tidl = params.inference.tidl;

    /*
      Models
    */

    {
      // Load TIDL parameters.
      read_tidl_model_configs(tidl.configs, false);

      // Load front model parameters.
      for (const std::string& model : tidl.configs.detector_models()) {
        read_tidl_config(                                        //
            tidl.configs.model_name_to_config(model).filename_,  //
            tidl.models.front);
      }

      // Load traffic model parameters.
      for (const std::string& model : tidl.configs.classifier_models()) {
        read_tidl_config(                                        //
            tidl.configs.model_name_to_config(model).filename_,  //
            tidl.models.traffic);
      }
    }

    /*
      Classes
    */

    {
      /*
        Object
      */

      // Classifications
      for (const TidlDecoderConfig& config : tidl.models.front.decoder_config) {
        struct {
          std::string name;
          ClassificationList ids;
        } cls;

        switch (config.type) {
          case TidlDecoderType::BBOX:
            cls.name = "bbox";
            break;

          case TidlDecoderType::VRUBOX:
          case TidlDecoderType::VRUBOX_W_ORIENTATION:
            cls.name = "vru";
            break;

          case TidlDecoderType::CUBOID:
            cls.name = "cuboid";
            break;

          case TidlDecoderType::CLASSIFIER:
          case TidlDecoderType::MULTI_CLASSIFIER:
          case TidlDecoderType::UNKNOWN:
          default:
            // Not supported!
            continue;
        }

        for (const std::string& name : config.classes) {
          cls.ids.push_back(phantomnet_object_name(name));
        }

        pv3.network.front.classes[cls.name].push_back(cls.ids);
      }

      // Segmentation
      {
        // Object Segmentation Map Classes
        const auto& classes = tidl.models.front.segmentation_config[0U].classes;

        // Make sure we have enough storage to store all mappings.
        if (classes.size() >  //
            utils::capacity(pv3.network.front.map)) {
          return Status::Failure;
        }

        // Store the mappings.
        for (uint32_t k = 0U; k < classes.size(); ++k) {
          pv3.network.front.map[k] = phantomnet_seg_name(classes[k]);
        }
      }

      /*
        Traffic
      */

      // Classifications
      for (const TidlDecoderConfig& config :
           tidl.models.traffic.decoder_config) {
        struct {
          const std::string name = "traffic";
          ClassificationList ids;
        } cls;

        for (const std::string& name : config.classes) {
          cls.ids.push_back(phantomnet_object_name(name));
        }

        pv3.network.traffic.classes[cls.name].push_back(cls.ids);
      }

      // Segmentation
      {
        // Traffic Segmentation Map Classes
        const auto& classes = tidl.models.front.segmentation_config[1U].classes;

        // Make sure we have enough storage to store all mappings.
        if (classes.size() >  //
            utils::capacity(pv3.network.traffic.map)) {
          return Status::Failure;
        }

        // Store the mappings.
        for (uint32_t k = 0U; k < classes.size(); ++k) {
          pv3.network.traffic.map[k] = phantomnet_seg_name(classes[k]);
        }
      }

      /*
        Visualization
      */

      {
        // Segmentation Map Configurations
        const auto& configs = tidl.models.front.segmentation_config;

        // Disable visualization by default.
        pv3.network.visualizer.id = -1;

        // Check if visualization is enabled.
        for (uint32_t id = 0U; id < configs.size(); ++id) {
          if (configs[id].enabled && configs[id].draw_viz) {
            pv3.network.visualizer.id = id;
            break;
          }
        }

        // If visualization is enabled
        if (pv3.network.visualizer.id >= 0) {
          // B, G, R
          const auto& colors = configs[pv3.network.visualizer.id].viz_colors;

          // Alpha
          const auto& alpha = configs[pv3.network.visualizer.id].blend_alpha;

          // Make sure we have enough storage to store all mappings.
          if ((alpha.size() > utils::capacity(pv3.network.visualizer.a)) ||
              (colors.size() > utils::capacity(pv3.network.visualizer.b)) ||
              (colors.size() > utils::capacity(pv3.network.visualizer.g)) ||
              (colors.size() > utils::capacity(pv3.network.visualizer.r))) {
            return Status::Failure;
          }

          // Store color mappings.
          for (uint32_t k = 0U; k < colors.size(); ++k) {
            if (colors[k].size() != 3U) {
              continue;
            }

            pv3.network.visualizer.b[k] = colors[k][0U];
            pv3.network.visualizer.g[k] = colors[k][1U];
            pv3.network.visualizer.r[k] = colors[k][2U];
          }

          // Store alpha mappings.
          for (uint32_t k = 0U; k < alpha.size(); ++k) {
            pv3.network.visualizer.a[k] =
                static_cast<uint8_t>(alpha[k] * 255.0F);
          }
        }
      }
    }

    /*
      Decoders
    */

    {
      using Object = Decode::Parameters::Object;
      auto& decode = params.inference.decode;

      for (const std::vector<float>& ratio :
           tidl.models.front.bbox_aspect_ratios) {
        decode.object.ratios.push_back({
            /* .x = */ ratio[0U],
            /* .y = */ ratio[1U],
        });
      }

      /* Main */
      for (const TidlDecoderConfig& config : tidl.models.front.decoder_config) {
        Object::Decoder decoder{
            /* .enabled = */ true,
            /* .type = */ convert(config.type),
            /* .id = */ cast::numeric<uint8_t>(config.id),
            /* .union = */ {},
        };

        switch (config.type) {
          case TidlDecoderType::BBOX:
          case TidlDecoderType::VRUBOX:
          case TidlDecoderType::VRUBOX_W_ORIENTATION: {
            decoder.box_or_vru = {
                /* .thresholds = */
                {
                    /* .data = */
                    config.thresholds.data(),

                    /* .count = */
                    cast::numeric<uint32_t>(config.thresholds.size()),
                },
            };
          } break;

          case TidlDecoderType::CUBOID: {
            decoder.cuboid = {
                /* .version = */ cast::numeric<uint8_t>(config.version),
                /* .classes = */
                {
                    /* .face = */
                    cast::numeric<uint8_t>(config.num_face_classes),

                    /* .side = */
                    cast::numeric<uint8_t>(config.num_side_classes),
                },
                /* .threshold = */
                {
                    /* .face = */ config.face_threshold,
                    /* .side = */ config.side_threshold,
                },
            };
          } break;

          case TidlDecoderType::CLASSIFIER:
          case TidlDecoderType::MULTI_CLASSIFIER:
          case TidlDecoderType::UNKNOWN:
          default:
            // Not supported!
            continue;
        }

        decode.object.decoders.push_back(decoder);
      }

      /* TSR */
      for (const TidlDecoderConfig& config : tidl.models.front.decoder_config) {
        if ("traffic_sign" == config.name) {
          break;
        }

        if (TidlDecoderType::BBOX == config.type) {
          ++decode.object.tsr;
        }
      }
    }

    /*
      PV3
    */

    params.init.inference = {
        /* .optimize = */ vision::Parameters::Optimize::Latency,
        /* .main = */
        {
            /* .model = */
            {
                /* .network = */ tidl.models.front.network_filename.c_str(),
                /* .config = */ tidl.models.front.config_filename.c_str(),
                /* .megaMACs = */ 1U,
            },
            /* .decoders = */
            {
                /* .center = */
                {
                    /* .enabled = */ tidl.models.front.enable_center_path,
                },
                /* .horizon = */
                {
                    /* .enabled = */ tidl.models.front.enable_horizon,
                },
                /* .object = */
                {
                    /* .enabled = */ true,
                    /* .anchor = */
                    {
                        /* .scale = */
                        {
                            /* .factor = */
                            cast::numeric<float>(
                                tidl.models.front.bbox_anchor_scale),

                            /* .count = */
                            cast::numeric<uint32_t>(
                                tidl.models.front.bbox_num_scales),
                        },
                        /* .ratios = */
                        {
                            /* .data = */
                            params.inference.decode.object.ratios.data(),

                            /* .count = */
                            cast::numeric<uint32_t>(
                                params.inference.decode.object.ratios.size()),
                        },
                    },
                    /* .decoders = */
                    {
                        /* .data = */
                        params.inference.decode.object.decoders.data(),

                        /* .count = */
                        cast::numeric<uint32_t>(
                            params.inference.decode.object.decoders.size()),
                    },
                    /* .pyramid = */
                    {
                        /* .min = */
                        cast::numeric<uint8_t>(  //
                            tidl.models.front.bbox_min_level),

                        /* .max = */
                        cast::numeric<uint8_t>(  //
                            tidl.models.front.bbox_max_level),

                        /* .start = */
                        cast::numeric<uint8_t>(  //
                            tidl.models.front.bbox_start_decoding_level),
                    },
                    /* .nms = */
                    {
                        /* .iou_threshold = */ 0.3F,
                    },
                },
                /* .segmentation = */
                {
                    /* .enabled = */ true,
                },
            },
            /* .callback = */
            std::bind(                         //
                callbacks::on_inference_main,  //
                std::ref(driver),              //
                std::placeholders::_1),
        },
        /* .tsr = */
        {
            /* .model = */
            {
                /* .network = */ tidl.models.traffic.network_filename.c_str(),
                /* .config = */ tidl.models.traffic.config_filename.c_str(),
                /* .megaMACs = */ 1U,
            },
            /* .decoder = */ params.inference.decode.object.tsr,
            /* .callback = */ {},
        },
    };
  }

  //
  // Image
  //

  {
    /*
      Convert
    */

    params.init.image.convert = {
        /* .optimize = */ vision::Parameters::Optimize::Latency,
        /* .format = */ vision::core::image::Format::iBGR,
        /* .callback = */
        std::bind(                  //
            callbacks::on_convert,  //
            std::ref(driver),       //
            std::placeholders::_1),
    };

    /*
      Pyramid
    */

    params.init.image.pyramid = {
        /* .optimize = */ vision::Parameters::Optimize::Latency,
        /* .levels = */ 2U,
        /* .kernel = */ 15U,
        /* .callback = */ {},
    };
  }

  //
  // Queue
  //

  {
    /*
      Input
    */

    // Image

    {
      // All access to the queue must be synchronized!
      Context::Queue::Input::Image& queue = pv3.queue.input.image;

      /* Critical Section */ {
        std::lock_guard lock(queue.mutex);

        queue.nv12.clear();
        queue.storage.clear();
      } /* Critical Section */
    }

    // Vehicle

    {
      // All access to the queue must be synchronized!
      Context::Queue::Input::Vehicle& queue = pv3.queue.input.vehicle;

      /* Critical Section */ {
        std::lock_guard lock(queue.mutex);

        queue.esp = EspMeasurements();
        queue.gs = GearState();
        queue.sa = SteeringAngle();
        queue.wsm = WheelSpeedMeasurements();
        queue.yaw = {};
      } /* Critical Section */
    }

    /*
      Output
    */

    // Convert

    {
      // All access to the queue must be synchronized!
      Context::Queue::Output::Convert& queue = pv3.queue.output.convert;

      /* Critical Section */ {
        std::lock_guard lock(queue.mutex);

        queue.quit = false;
        queue.frame = 0U;
        queue.rgb.clear();

        // Launch the background worker thread.
        queue.worker = std::thread(workers::convert, std::ref(driver));
      } /* Critical Section */
    }

    // Decode

    {
      // All access to the queue must be synchronized!
      Context::Queue::Output::Decode& queue = pv3.queue.output.decode;

      /* Critical Section */ {
        std::lock_guard lock(queue.mutex);

        queue.quit = false;
        queue.frame = 0U;
        queue.data.status = Status::Failure;

        // Launch the background worker thread.
        queue.worker = std::thread(workers::inference, std::ref(driver));
      } /* Critical Section */
    }
  }

  pv3.context = {};
  return vision::initialize(pv3.context, params.init);
}

Status
shutdown(driver::Context& driver) {
  // Alias
  pv3::Context& pv3 = driver.pv3;

  {
    // All access to the queue must be synchronized!
    Context::Queue& queue = pv3.queue;

    /* Critical Section */ {
      std::lock_guard lock(queue.output.convert.mutex);
      queue.output.convert.quit = true;
    } /* Critical Section */

    /* Critical Section */ {
      std::lock_guard lock(queue.output.decode.mutex);
      queue.output.decode.quit = true;
    } /* Critical Section */

    // Notify the worker threads of our termination request.
    queue.output.convert.produced.notify_all();
    queue.output.decode.produced.notify_all();

    // Join the worker threads.
    queue.output.convert.worker.join();
    queue.output.decode.worker.join();
  }

  return vision::shutdown(pv3.context);
}

}  // namespace pv3
}  // namespace driver
}  // namespace phantom_ai
