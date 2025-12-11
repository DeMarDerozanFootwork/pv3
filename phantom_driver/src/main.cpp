/*******************************************************************************
 * @file    main.cpp
 * @date    10/2024
 *
 * @attention Copyright (c) 2022-2024
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

// HAL
#include "hal_camera.h"
#include "tidl_kernels.h"

// Phantom Vision 2
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include <phantom_ai/core/profiler.h>
#include <phantom_ai/phantom_vision2/camera_motion_task.h>
#include <phantom_ai/phantom_vision2/feature_track_task.h>
#include <phantom_ai/phantom_vision2/lane_task.h>
#include <phantom_ai/phantom_vision2/object_task.h>
#include <phantom_ai/phantom_vision2/visualizer_task.h>
#include <phantom_ai/vehicle_state_parser/vehicle_state_parser.h>
#include <phantom_ai/wrappers/camera_wrapper.h>
#include <phantom_ai/wrappers/can_wrapper.h>
#include <phantom_ai/wrappers/network_rx_wrapper.h>
#pragma GCC diagnostic pop

// Phantom Vision 3
#include <phantom_ai/phantom_vision3/perception.h>

// CRT / STL
#include <csignal>

namespace phantom_ai {
namespace {

// Alias
using Status = vision::Status;
using Clock = std::chrono::high_resolution_clock;

//
// Driver
//

struct Driver final {
  enum class Input : uint8_t {
    HAL,
    Network,
  };

  struct Config final {
    static constexpr Input kInput = Input::Network;
  };

  struct PV2 final {
    // Thread-safety: Strictly read-only at runtime.
    VisionParams params;

    struct /* Thread-safety: Strictly read-only at runtime. */ {
      CameraModelListS models;
      std::vector<camera_params_hal> params;
      std::vector<CameraID> views;
      std::unordered_map<uint32_t /* PV3 View Index */, CameraID> map;
      std::unique_ptr<CameraWrapper> wrapper;
    } camera;

    struct /* Thread-safety: Strictly read-only at runtime. */ {
      std::unique_ptr<CanWrapper> bus0;
      std::unique_ptr<CanWrapper> bus1;
      std::unique_ptr<VehicleStateParser> parser;
    } can;

    struct /* Thread-safety: Strictly read-only at runtime. */ {
      std::unique_ptr<NetRxWrapper> wrapper;
    } network;

    // Thread-safety: PV2 MessageQueue is thread-safe.
    std::array<MessageQueue, NUM_VISION_MESSAGES> messages;
    std::vector<std::unique_ptr<VisionTaskBase>> tasks;
  } pv2;

  struct PV3 final {
    vision::Context context;
    vision::Output output;

    struct /* Thread-safety: Strictly read-only at runtime. */ {
      std::vector<vision::camera::View> views;
      std::unordered_map<CameraID, uint32_t /* PV3 View Index */> map;
    } camera;

    struct /* Thread-safety: Strictly read-only at runtime. */ {
      std::unordered_map<std::string, std::vector<ClassificationList>> classes;
    } network;

    // Thread-safety: All access must be explicitly synchronized.
    struct Queue final {
      struct Input final {
        struct Image final {
          std::mutex mutex;
          std::condition_variable produced;
          std::vector<cv::Mat> storage;
          std::vector<vision::image::Handle> nv12;
        } image;

        struct Vehicle final {
          std::mutex mutex;
          EspMeasurements esp;
          WheelSpeedMeasurements wsm;
          SteeringAngle sa;
          GearState gs;
          struct {
            float rate;
            float bias;
            float count;
          } yaw;
        } vehicle;
      } input;

      struct Output final {
        struct Convert final {
          std::thread worker;
          std::mutex mutex;
          std::condition_variable produced;
          std::vector<vision::image::Handle> rgb;
          uint32_t frame;
          bool quit;
        } convert;

        struct Decode final {
          std::thread worker;
          std::mutex mutex;
          std::condition_variable produced;
          vision::Output::Inference::Pipeline::Decode data;
          uint32_t frame;
          bool quit;
        } decode;
      } output;
    } queue;
  } pv3;
};

//
// API
//

Status initialize(Driver& driver) noexcept;
Status process(Driver& driver);
Status shutdown(Driver& driver) noexcept;

}  // namespace
}  // namespace phantom_ai

int
main() {
  // Phantom AI
  using namespace phantom_ai;

  Driver driver{};
  Status status{};

  //
  // Initialize
  //

  if (Status::Success != initialize(driver)) {
    std::cerr << "Failed to initialize Phantom Driver!" << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Process
  //

  do {
    // Execute!
    status = process(driver);
  } while (Status::Success == status);

  //
  // Shutdown
  //

  if (Status::Success != shutdown(driver)) {
    std::cerr << "Failed to deinitialize Phantom Driver!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

//
// RUNTIME
//

namespace phantom_ai {
namespace {

// Signal
volatile std::sig_atomic_t terminate{};

namespace pv3 {

// Alias
namespace cast = vision::core::type::cast;

// Decode
struct Decode final {
  // PV3 compile-time inference decoding configuration parameters
  using Configuration = vision::core::Configuration::Inference::Decode;

  // PV3 initialization-time inference decoding configuration parameters
  using Parameters = vision::Parameters::Inference::Pipeline::Decode;

  // PV3 per-frame inference decoding output
  using Output = vision::Output::Inference::Pipeline::Decode;
};

// Declarations
vision::camera::View convert(CameraID location) noexcept;
vision::core::image::Format convert(hal_camera_format format) noexcept;
vision::image::Handle convert(cv::Mat image) noexcept;
Decode::Parameters::Object::Type convert(TidlDecoderType type) noexcept;

}  // namespace pv3

//
// IMPORTANT NOTE!
//
// This function MUST BE kept as light as possible!  Do NOT perform any extra
// processing here except for the absolute bare minimum!  Ideally, that simply
// takes the form of calling PV3 in a loop and nothing else!  Keep in mind that
// PV3 is the front end of the vision pipeline, and produces data that the rest
// of the perception stack has a blocking dependency on.  Furthermore, PV3 runs
// inference which currently is the performance bottleneck in our pipeline.
// Because of these two reasons, the rate at which PV3 runs places a ceiling on
// the performance of the entirety of the vision system.  Any slowdowns here,
// introduced as a result of any extra processing performed, will impact the
// performance of the entirety of the vision pipeline!

Status
process(Driver& driver) {
  Driver::PV3& pv3 = driver.pv3;

  if (terminate) {
    std::cout << "Received signal (" << terminate << "). Terminating ...\n";
    return Status::Done;
  }

  /* Critical Section */ {
    // All acccess to the queue must be synchronized!
    Driver::PV3::Queue::Input::Image& queue = pv3.queue.input.image;

    // Input Queue Synchronization:
    // In case PV3 is not capturing its camera output internally, and instead is
    // relying on externally captured images provided to it as input, we need to
    // make sure that the said input images will stay valid and intact for as
    // long as they are being processed by PV3.  Since these externally acquired
    // images are produced by a separate thread running the on_pv2_capture_*
    // callback in parallel, we need to be careful with data races!  Here, we
    // lock the input queue mutex to make sure that the input images it is
    // protecting will not be overwritten while they are being copied over into
    // PV3's internal buffers.  We will unlock this mutex as soon as PV3 is done
    // with this copying, which would be when the on_pv3_capture callback is
    // called.  As an aside, keep in mind that using HAL as the camera capture
    // backend is not ideal for performance as it incurs two extra memcpy's per
    // image per frame: one set from HAL's internal buffers to a user provided
    // memory, and a second set from the said intermediary region of user memory
    // into PV3's internal buffers.  Finally, this locking mechanism may and
    // should be avoided when using PV3 in its internal capture mode.
    std::unique_lock<std::mutex> lock(queue.mutex);

    // With the mutex locked, wait until an image is available.  Note that the
    // condition variable will atomically unlock the lock and block this thread
    // until the condition is met, at which point this thread is unblocked, and
    // the lock re-locked prior to the function returning.
    queue.produced.wait(lock, [&queue]() noexcept {
      return !queue.storage.empty() &&  //
             !queue.nv12.empty();
    });

    // As mentioned, the mutex will be locked after condition_variable::wait()
    // returns.  We would like to keep the mutex locked as mentioned in the
    // comment above in order to keep the data intact just until PV3 is done
    // with it, which would be when on_pv3_capture is called.  At that point, we
    // will manually unlock the mutex.  But since unique_lock unlocks the mutex
    // in its destructor, called at the end of this scope, we disengage the lock
    // to leave the mutex in a locked state.
    lock.release();

    const vision::Input input{
        /* .camera = */
        {
            /* .capture = */
            {
                /* .views = */
                {
                    /* .data = */
                    pv3.camera.views.data(),

                    /* .count = */
                    pv3::cast::numeric<uint32_t>(pv3.camera.views.size()),
                },

                /* .images = */
                {
                    /* .data = */
                    queue.nv12.data(),

                    /* .count = */
                    pv3::cast::numeric<uint32_t>(queue.nv12.size()),
                },
            },
        },
    };

    // Call PV3!  Keep in mind that PV3 will actually not run in a critical
    // section for the entirety of the time vision::process() is executing
    // as we will unlock the mutex after PV3 is done with its first stage which
    // would be when the on_pv3_capture callback is called.
    return vision::process(pv3.context, input, pv3.output);
  } /* Critical Section */
}

/*
  PV2 Callbacks
*/

void
on_pv2_capture_hal(  //
    Driver& driver,  //
    std::vector<CameraResults>& images) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // PV3 (read-only)
  const Driver::PV3& pv3 = driver.pv3;

  // All acccess to the queue must be synchronized!
  Driver::PV3::Queue::Input::Image& queue = driver.pv3.queue.input.image;

  // Variable to track when to notify the consumer.  We only do so when new
  // images are available and only then if all received images are valid.
  uint32_t received = 0U;

  /* Critical Section */ {
    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // We have the following two options in dealing with this queue: (1) either
    // block further production until all previously produced data is consumed
    // by the main thread and published to its corresponding PV2 tasks, or (2)
    // drop data not yet consumed and replace it with a more recent batch.  This
    // implementation chooses the 2nd approach.  Note that there will be no
    // perceptible difference between the two options if consumption keeps up
    // with production at all times.  Otherwise, the first approach may
    // intorduce some jitter but will have the advantage of the system gathering
    // and processing more information from the environment.

    // Clear the queue in case it is not yet consumed which would happen if we
    // keep producing output faster than the main thread can process.
    queue.storage.clear();
    queue.nv12.clear();

    // Grow the vectors if needed to allocate space for all expected views.
    queue.storage.resize(pv3.camera.map.size());
    queue.nv12.resize(pv3.camera.map.size());

    // For each captured image (which may or may not come from the same camera)
    for (uint32_t image = 0U; image < images.size(); ++image) {
      // Get the metadata associated with the current image.
      const CameraResults& result = images[image];

      // If the image is valid
      if (const auto itr = pv3.camera.map.find(result.id);
          (pv3.camera.map.cend() != itr) && !result.image.empty()) {
        // Get the corresponding PV3 view index.
        const uint32_t view = itr->second;

        // Given HalCamera's current implementation, cv::Mat
        // CameraResults::image owns its memory.  The object's memory is
        // allocated on the heap as part of cv::Mat's constructor and is owned
        // and managed internally by cv::Mat itself.  As a result, as the long
        // as the object is kept alive, its backing memory will be valid as
        // well.  Furthermore, cv::Mat's internal memory is reference counted
        // and cv::Mat is designed to operate on the basis of shared memory
        // ownership.  Copies performed through the assignment operator, like
        // the one below, are shallow, instead simply bumping up a shared
        // refcount, in turn keeping the memory alive.  This operation does not
        // copy any substansive amount of data.
        queue.storage[view] = result.image;

        // With the matrix kept alive, convert PV2 cv::Mat to PV3
        // image::Handles. Note that PV3 image::Handles are thin objects that
        // simply describe, and point to, a location in memory and do not own
        // the image data and its lifetime by design, hence, why we needed to
        // keep the matrix alive as the object in charge of the image memory.
        // This operation does not copy any substansive amount of data.
        queue.nv12[view] = pv3::convert(result.image);

        // Increment the number of valid images received.
        ++received;
      }
    }
  } /* Critical Section */

  // Only if we have received a valid image for each configured view
  if (driver.pv3.camera.views.size() == received) {
    // Notify the main thread that new images are ready.
    queue.produced.notify_one();

    // Relinquishing the remainder of this thread's time slice is not at all
    // necessary for functional correctness but profiling shows that doing so
    // sometimes helps with performance knowing how the main thread running PV3
    // and inference (a) has a blocking dependency on this data, and (b) is the
    // execution bottleneck in our system so the sooner it's woken up and
    // scheduled to run, the better.  Still, keep in mind that this is just a
    // hint and a shot in the dark at best and the operating system is under no
    // obligation whatsoever to actually choose the main thread out of the many
    // threads waiting as the next thread to assign to this core but still this
    // thread is low priority so almost any other task is more urgent.
    sched_yield();
  }
}

void
on_pv2_capture_net(  //
    Driver& driver,  //
    const Embedded_Frame& frame) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // PV3 (read-only)
  const Driver::PV3& pv3 = driver.pv3;

  /*
    Capture
  */

  // Variable to track when to notify the consumer.  We only do so when new
  // images are available and only then if all received images are valid.
  uint32_t received = 0U;

  /* Critical Section */ {
    // All acccess to the queue must be synchronized!
    Driver::PV3::Queue::Input::Image& queue = driver.pv3.queue.input.image;

    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // Pass 1:
    // We have the following two options in dealing with this queue: (1) either
    // block further production until all previously produced data is consumed
    // by the main thread and published to its corresponding PV2 tasks, or (2)
    // drop data not yet consumed and replace it with a more recent batch.  This
    // implementation chooses the 2nd approach.  Note that there will be no
    // perceptible difference between the two options if consumption keeps up
    // with production at all times.  Otherwise, the first approach may
    // intorduce some jitter but will have the advantage of the system gathering
    // and processing more information from the environment.  Here, we scan the
    // received frames to see if we are dealing with any images, and only if so,
    // proceed to clear the queue.

    for (uint32_t entry = 0U; entry < frame.header.item_count; ++entry) {
      // Get the metadata associated with the current entry.
      const EmbeddedNet_PhantomHeader_Info& item = frame.header.item[entry];

      // In case we are dealing with an image
      if ((CAMERA_IMG_BGR == item.format) ||  //
          (CAMERA_IMG_RGB == item.format) ||  //
          (CAMERA_IMG_YUV_420SP == item.format)) {
        // Clear the queue in case it is not yet consumed which would happen if
        // we keep producing faster than the main thread can process.
        queue.storage.clear();
        queue.nv12.clear();

        // Grow the vectors if needed to allocate space for all expected views.
        queue.storage.resize(pv3.camera.map.size());
        queue.nv12.resize(pv3.camera.map.size());

        // Clearing the queue only needs to be done once; break.  We only needed
        // to make sure that this Embedded_Frame indeed contained image data
        // prior to clearing the queue.
        break;
      }
      else {
        // Not an image!  Do not handle any non-image cases here in this CS.
      }
    }

    // Pass 2:
    // PV3 only accepts NV12 images! Convert if necessary.

    for (uint32_t entry = 0U; entry < frame.header.item_count; ++entry) {
      // Get the metadata associated with the current entry.
      const EmbeddedNet_PhantomHeader_Info& item = frame.header.item[entry];

      // Find the corresponding PV3 View Index to this PV2 CameraID.
      const auto itr = pv3.camera.map.find(static_cast<CameraID>(item.id));

      // Skip if view is not in use.
      if (pv3.camera.map.cend() == itr) {
        continue;
      }

      // Get the corresponding PV3 view index.
      const uint32_t view = itr->second;

      // In case we are dealing with an image
      if ((CAMERA_IMG_BGR == item.format) ||  //
          (CAMERA_IMG_RGB == item.format)) {
        // A conversion to NV12 is necessary when input is BGR / RGB.
        const struct {
          uint32_t offset;  // Bytes
          uint32_t size;    // Bytes
        } src{
            /* .offset = */ item.offset[/* Plane = */ 0U],  // Interleaved
            /* .size = */ item.buf_size[/* Plane = */ 0U],  // Interleaved
        };

        // If the image metadata is valid
        if ((src.offset + src.size) < frame.data_size) {
          // Get the pointer to the input image.
          uint8_t* const ptr = frame.data + src.offset;

          // If the image is valid
          if (ptr && (item.height > 0U) && (item.width > 0U)) {
            // Create a new OpenCV matrix to wrap around the captured iBGR /
            // iRGB image.  This operation will not allocate any memory.
            const cv::Mat ibgr(item.height, item.width, CV_8UC3, ptr);

            // PV3 only accepts NV12 images as input as of now.
            cv::Mat nv12((item.height * 3U) / 2U, item.width, CV_8UC1);

            // We handle both BGR and RGB in this code path.
            const cv::ColorConversionCodes code =  //
                (CAMERA_IMG_BGR == item.format)    //
                    ? cv::COLOR_BGR2YUV_I420
                    : cv::COLOR_RGB2YUV_I420;

            // Convert BGR / RGB -> I420.  OpenCV does not support a direct
            // conversion to NV12.  OpenCV's conversions, not to mention our
            // additional post processing on top of it below, are not well
            // optimized, but profiling shows this is not a major bottleneck for
            // rosbags.  Still, not ideal but will do for rosbags.
            cv::cvtColor(ibgr, nv12, code);

            const struct {
              uint32_t y;
              uint32_t u;
              uint32_t uv;
            } bytes{
                /* .y = */ item.height * item.width,
                /* .u = */ bytes.y / 4U,
                /* .uv = */ bytes.u * 2U,
            };

            std::vector<uint8_t> uv(bytes.uv);
            std::copy(nv12.data + bytes.y,             //
                      nv12.data + bytes.y + bytes.uv,  //
                      uv.data());

            // Interleave
            for (uint32_t i = 0U; i < bytes.u; i++) {
              nv12.data[bytes.y + 2U * i] = uv[i];                 // U
              nv12.data[bytes.y + 2U * i + 1U] = uv[bytes.u + i];  // V
            }

            // Add the newly received images to the queue.
            queue.storage[view] = nv12;
            queue.nv12[view] = pv3::convert(nv12);

            // Increment the number of valid images received.
            ++received;
          }
        }
      }
      else if (CAMERA_IMG_YUV_420SP == item.format) {
        // NV12 conversion not necessary.  TODO: memcpy.
      }
      else {
        // Not an image!  Do not handle any non-image cases here in this CS.
      }
    }
  } /* Critical Section */

  /*
    Vehicle State
  */

  /* Critical Section */ {
    // All acccess to the queue must be synchronized!
    Driver::PV3::Queue::Input::Vehicle& queue = driver.pv3.queue.input.vehicle;

    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // For each new entry
    for (uint32_t entry = 0U; entry < frame.header.item_count; ++entry) {
      // Get the metadata associated with the current entry.
      const EmbeddedNet_PhantomHeader_Info& item = frame.header.item[entry];

      // Only if we are dealing with vehicle state

      if (VEHICLE_STATE == item.format) {
        const EmbeddedNet_VehicleInfo& vehicle =
            reinterpret_cast<const EmbeddedNet_VehicleInfo&>(
                frame.data[item.offset[0U]]);
        //
        // Note: This code is taken from VisionNetworkApp2::processFrame()
        //

        // We need to match up wheel speed and yaw rate measurments.  The two
        // measurements are not synced; in Genesis car the yaw rate is 2x the
        // wheel speed.  Will need to duplicate some wheel speed measurements to
        // "fill" in data.

        const struct {
          float esp;
          float wsm;
        } ratio{
            /* .esp = */
            [&vehicle]() noexcept {
              if (vehicle.wheelspeed_count <= vehicle.esp_count) {
                return 1.0F;
              }

              if (vehicle.wheelspeed_count > 0U) {
                return pv3::cast::numeric<float>(vehicle.esp_count) /
                       pv3::cast::numeric<float>(vehicle.wheelspeed_count);
              }

              return 0.0F;
            }(),

            /* .wsm = */
            [&vehicle]() noexcept {
              if (vehicle.esp_count < vehicle.wheelspeed_count) {
                return 1.0F;
              }

              if (vehicle.esp_count > 0U) {
                return pv3::cast::numeric<float>(vehicle.wheelspeed_count) /
                       pv3::cast::numeric<float>(vehicle.esp_count);
              }

              return 0.0F;
            }(),
        };

        const uint32_t count = std::max(  //
            vehicle.wheelspeed_count,     //
            vehicle.esp_count);

        // Question: Each iteration of this loop continously overwrites values
        // calculated in previous iterations.  Is this the intended behavior?!
        for (uint32_t i = 0U; i < count; ++i) {
          const struct {
            uint32_t esp;
            uint32_t wsm;
          } index{
              /* .esp = */ pv3::cast::numeric<uint32_t>(i * ratio.esp),
              /* .wsm = */ pv3::cast::numeric<uint32_t>(i * ratio.wsm),
          };

          if (index.esp < vehicle.esp_count) {
            convertEsp(vehicle.esp[index.esp], queue.esp);
          }

          if (index.wsm < vehicle.wheelspeed_count) {
            convertWheelSpeed(vehicle.wheelspeed[index.wsm], queue.wsm);
          }
        }

        //
        // Note: This code is taken from VisionWrapper2::calculateYawRateBias()
        //

        {
          constexpr float kMinNumSamples = 50.0F;
          constexpr float kMaxYawRateCnt = 10000.0F;
          constexpr float kStationaryVehicleSpeedThresholdKph = 1.0F;
          constexpr float kYawRateThresholdInStationarySituationDeg = 1.5F;
          float wsm = 0.5F * (queue.wsm.rear_left_ + queue.wsm.rear_right_);

          if (wsm < kStationaryVehicleSpeedThresholdKph) {
            if (std::abs(queue.esp.yaw_rate_) <
                kYawRateThresholdInStationarySituationDeg) {
              queue.yaw.bias =
                  ((queue.yaw.bias * queue.yaw.count) + queue.esp.yaw_rate_) /
                  (queue.yaw.count + 1.0F);

              queue.yaw.count = std::min(  //
                  queue.yaw.count + 1.0F,  //
                  kMaxYawRateCnt);
            }
          }

          queue.yaw.rate = (queue.yaw.count > kMinNumSamples)  //
                               ? queue.yaw.bias
                               : 0.0F;
        }
      }
      else {
        // Not a vehicle state!  Do not handle any other cases here in this CS.
      }
    }
  } /* Critical Section */

  // If we received the correct number of images
  if (driver.pv3.camera.views.size() == received) {
    // Notify the main thread that new images are ready.
    driver.pv3.queue.input.image.produced.notify_one();

    // Relinquishing the remainder of this thread's time slice is not at all
    // necessary for functional correctness but profiling shows that doing so
    // sometimes helps with performance knowing how the main thread running PV3
    // and inference (a) has a blocking dependency on this data, and (b) is the
    // execution bottleneck in our system so the sooner it's woken up and
    // scheduled to run, the better.  Still, keep in mind that this is just a
    // hint and a shot in the dark at best and the operating system is under no
    // obligation whatsoever to actually choose the main thread out of the many
    // threads waiting as the next thread to assign to this core but still this
    // thread is low priority so almost any other task is more urgent.
    sched_yield();
  }
}

/*
  PV3 Callbacks
*/

//
// Note: Do not perform any computationally intensive operations on any of the
// below PV3 callbacks!  These callbacks must be kept as light as possible with
// any extra processing scheduled for a later time on a worker thread.  Think of
// Linux ISR top-halves and bottom-halves with these callbacks as the former.
//

void
on_pv3_capture(      //
    Driver& driver,  //
    const vision::Output::Camera::Capture& /*capture*/) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // Note 1: This code is not necessary if PV3 is run in internal capture mode.
  // Note 2: We leave vehicle state data intact.

  /* Critical Section */ {
    // All acccess to the queue must be synchronized!
    Driver::PV3::Queue::Input::Image& queue = driver.pv3.queue.input.image;

    // Clear the queue.
    queue.nv12.clear();
    queue.storage.clear();

    // Refer to the comment titled 'Input Queue Synchronization' in process().
    queue.mutex.unlock();
  } /* Critical Section */
}

void
on_pv3_convert(      //
    Driver& driver,  //
    const vision::Output::Image::Convert& convert) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // We currently have no use for the converted grayscale images.  Technically,
  // we could put the grayscale images to use if we tried hard enough but
  // assuming that we will replace PV2 feature tracking with that of PV3 soon
  // enough, that would be more effort than is worth it.  As an aside, using PV2
  // feature tracking means we will be doing an unnecessary grayscale color
  // conversion.  PV3 reuses the luma channel of the input NV12 image for free.

  // Don't bother if RGB color conversion failed for whatever reason.
  if (Status::Success != convert.rgb.status) {
    return;
  }

  // All acccess to the queue must be synchronized!
  Driver::PV3::Queue::Output::Convert& queue = driver.pv3.queue.output.convert;

  /* Critical Section */ {
    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // We have the following two options in dealing with this queue: (1) either
    // block further production until all previously produced data is consumed
    // by the worker thread and published to its corresponding PV2 tasks, or (2)
    // drop data not yet consumed and replace it with a more recent batch.  This
    // implementation chooses the 2nd approach.  Note that there will be no
    // perceptible difference between the two options if consumption keeps up
    // with production at all times.  Otherwise, the first approach may
    // intorduce some jitter but will have the advantage of the system gathering
    // and processing more information from the environment.

    // Clear the queue in case it is not yet consumed which would happen if we
    // keep producing output faster than the background thread can process.
    queue.rgb.clear();

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
on_pv3_inference(    //
    Driver& driver,  //
    const vision::Output::Inference::Pipeline& inference) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // We are only interested in decoding results for now.
  const pv3::Decode::Output& decode = inference.decode;

  // Don't bother if inference decoding failed for whatever reason.
  if (Status::Success != decode.status) {
    return;
  }

  // All acccess to the queue must be synchronized!
  Driver::PV3::Queue::Output::Decode& queue = driver.pv3.queue.output.decode;

  /* Critical Section */ {
    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // We have the following two options in dealing with this queue: (1) either
    // block further production until all previously produced data is consumed
    // by the worker thread and published to its corresponding PV2 tasks, or (2)
    // drop data not yet consumed and replace it with a more recent batch.  This
    // implementation chooses the 2nd approach.  Note that there will be no
    // perceptible difference between the two options if consumption keeps up
    // with production at all times.  Otherwise, the first approach may
    // intorduce some jitter but will have the advantage of the system gathering
    // and processing more information from the environment.

    // Override the queue contents even if its contents are yet to be consumed
    // by the worker thread which would happen if we keep producing output
    // faster than the background thread can process.  This operation does not
    // copy any substansive amount of data.  Remember that our goal with PV3
    // callbacks is to return as soon as possible.
    queue.data = decode;

    // Track the frame numbers this data corresponds to.
    queue.frame = decode.frames.front();
  } /* Critical Section */

  // Notify the worker thread that a new output is ready.
  queue.produced.notify_one();

  // Note: Do not yield.  This function is on the critical path.
}

/*
  PV3 Worker Threads
*/

//
// Note: It is safer to perform computationally intenstive operations on these
// background worker threads.  Still, keep in mind that there is no free lunch
// in computing in that these threads will still be competing with the rest of
// the system for CPU time.  Try to keep things light!  Following our Linux ISR
// top-halves and bottom-halves analogy, these workers will be the latter.
//

void
pv3_convert_worker(Driver& driver) {
  Driver::PV2& pv2 = driver.pv2;
  Driver::PV3& pv3 = driver.pv3;

  while (true) {
    CamerasDataS camera{};
    VehicleStateDataS vsd{};

    /* Critical Section */ {
      // All acccess to the queue must be synchronized!
      Driver::PV3::Queue::Output::Convert& queue = pv3.queue.output.convert;

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

      // Set metadata.
      camera->frame() = queue.frame;
      camera->t_hw() = T_NOW;

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
        const vision::image::Handle image = queue.rgb[view];

        // Create a new OpenCV matrix.  This will dynamically allocate memory.
        cv::Mat matrix(image.height, image.width, CV_8UC3);

        // Copy the converted image over.  These copies, plus the corresponding
        // set in HAL, will be avoided when using PV3 in internal capture mode.
        std::memcpy(                 //
            matrix.ptr(),            //
            image.data.as_void_ptr,  //
            matrix.total() * matrix.elemSize());

        // Add the image to its PV2 container.
        camera->Insert(pv2.camera.map[view], T_NOW_SEC, false, matrix);
      }

      // Done!
      queue.rgb.clear();
    } /* Critical Section */

    // Post & Draw
    if (camera) {
      // Post the captured images to all interested parties. Note that PV2's
      // message queue is thread-safe.  No extra locking mechanism is required.
      pv2.messages[MESSAGE_CAMERAS].AddSharedData(camera);

      /* Critical Section */ {
        // All acccess to the queue must be synchronized!
        Driver::PV3::Queue::Input::Vehicle& queue = pv3.queue.input.vehicle;

        // We cannot modify the queue if its contents are modified in parallel.
        std::unique_lock<std::mutex> lock(queue.mutex);

        // Create a new PV2 payload for this data.
        vsd = std::make_shared<VehicleStateData>(         //
            camera->t_hw(),                               //
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
        pv2.messages[MESSAGE_VEHICLE_STATE_ROS].AddSharedData(vsd);
      } /* Critical Section */

      // Draw only if visualization is enabled.  Assuming params is read-only.
      if (pv2.params.visualizer_task.enable_task) {
        VisualizerDataS canvas = std::make_shared<VisualizerData>(camera);
        canvas->Initialize(pv2.params.visualizer_window);
        canvas->Draw(camera, VIZ_DRAW_CAMERAS_COLOR);
        canvas->mode() = 1;
        pv2.messages[MESSAGE_VISUALIZER].AddSharedData(canvas);
      }
    }
  }
}

void
pv3_inference_worker(Driver& driver) {
  Driver::PV2& pv2 = driver.pv2;
  Driver::PV3& pv3 = driver.pv3;

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
      // All acccess to the queue must be synchronized!
      Driver::PV3::Queue::Output::Decode& queue = pv3.queue.output.decode;

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
          pv2.messages[MESSAGE_CAMERAS],

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
                  /*. main = */ pv3.network.classes["bbox"],
                  /*. traffic = */ pv3.network.classes["traffic"],
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
                  /*. main = */ pv3.network.classes["vru"],
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

            // Copy the images over.  We could technically avoid these memory
            // copies and instead use the internal PV3 memory pointed to by
            // these handles directly if we could somehow guarantee that the
            // created segmentation masks would have been completely consumed
            // and destroyed within N frames, where N is the broadcasted
            // lifetime of the data pointed to by these handles, but since we
            // cannot easily guarantee that with certainty given PV2's reliance
            // on shared ownership semantics (i.e. the object may stay alive for
            // an indeterminate period of time after it's been created and sent
            // out as long as someone, somewhere, holds a reference to it, even
            // accidentally), we must perform a deep copy to be safe.

            // We assume the mask has the same dimensions as the original image.
            const cv::Size size = payload->image_size();

            // Allocate memory for the object and lane segmentation masks.
            cv::Mat object(size.height, size.width, CV_8UC1);
            cv::Mat lane(size.height, size.width, CV_8UC1);

            // Copy the object segmentation mask.
            std::memcpy(                   //
                object.ptr(),              //
                src[0U].data.as_void_ptr,  //
                object.total() * object.elemSize());

            // Copy the lane segmentation mask.
            std::memcpy(                   //
                lane.ptr(),                //
                src[1U].data.as_void_ptr,  //
                lane.total() * lane.elemSize());

            // Pass on the matrices.
            payload->seg() = object;
            payload->lane_seg() = lane;
          }

          // TODO: I suspect this usage is wrong!
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
      // Post the captured images to all interested parties. Note that PV2's
      // message queue is thread-safe.  No extra locking mechanism is required.
      pv2.messages[MESSAGE_PHANTOMNET_A].AddSharedData(payload);

      // Draw only if visualization is enabled.  Assuming params is read-only.
      if (pv2.params.visualizer_task.enable_task) {
        struct {
          MessageQueue& queue;
          const VisualizerDataS payload;
        } visualizer{
            /* .queue = */
            pv2.messages[MESSAGE_VISUALIZER],

            /* .payload = */
            visualizer.queue.PeekMessageFrameAt<VisualizerData>(  //
                TASK_VISUALIZER,                                  //
                payload->frame()),
        };

        // If found a viz message for this frame,
        if (visualizer.payload) {
          // Draw!
          visualizer.payload->Draw(                      //
              payload,                                   //
              VIZ_DRAW_PHANTOMNET_CUBOID |               //
                  /* VIZ_DRAW_PHANTOMNET_DETECTION | */  //
                  VIZ_DRAW_PHANTOMNET_OBJECT |           //
                  VIZ_DRAW_PHANTOMNET_HORIZON_DIST);

          // Post the captured images to the queue. Note that PV2's message
          // queue is thread-safe.  No extra locking mechanism is required.
          pv2.messages[MESSAGE_VISUALIZER].AddSharedData(visualizer.payload);
        }
      }
    }
  }
}

}  // namespace
}  // namespace phantom_ai

//
// BOILERPLATE
//

namespace phantom_ai {
namespace {

//
// Phantom Vision 2
//

namespace pv2 {

template <typename Task>
std::unique_ptr<Task>
create_task(Driver::PV2& pv2, const int frequency) {
  std::unique_ptr<Task> task = std::make_unique<Task>(pv2.messages);
  task->RegisterCameraModelList(pv2.camera.models);
  task->RegisterMainTaskParam(pv2.params.main_task);
  task->RegisterVisualizerTaskParam(pv2.params.visualizer_task);
  task->RegisterVisualizerWindowParam(pv2.params.visualizer_window);
  task->SetTimer(frequency);
  return task;
}

Status
initialize(Driver& driver) {
  Driver::PV2& pv2 = driver.pv2;

  //
  // Configuration
  //

  {
    const YamlNode yaml = load_yaml_file(    //
        "perception/phantom_vision2/tda4x",  //
        "vision_param2_tda4x_front_2mp_3crops.yaml");

    if (!yaml) {
      return Status::Failure;
    }

    if (!read_vision_params(yaml, pv2.params)) {
      return Status::Failure;
    }
  }

  //
  // Camera
  //

  {
    /*
      Models
    */

    pv2.camera.models = std::make_shared<CameraModelList>(NUM_MAX_CAM_IDS);
    pv2.camera.models->Initialize(                                            //
        pv2.params.camera_model_param_file,                                   //
        pv2.params.vehicle_name,                                              //
        pv2.params.target_system,                                             //
        pv2.params.camera_image_size,                                         //
        merge_cameras(pv2.params.cameras_active, pv2.params.cameras_remote),  //
        pv2.params.cameras_image_mask,                                        //
        PhantomVisionDynamicCalibrationInput(),                               //
        pv2.params.ros_node.verbosity & VERBOSITY_MAIN_CAMERA_CALIB);

    /*
      Parameters
    */

    std::vector<int32_t> annotation;
    CameraInfoList info;

    readHalCamerasParameters(        //
        "front_low_res_15fps.yaml",  //
        pv2.camera.params,           //
        info,                        //
        annotation,                  //
        false,                       //
        false,                       //
        pv2.params.target_system);

    /*
      Views
    */

    // For each detected HAL camera
    for (const camera_params_hal& param : pv2.camera.params) {
      // For each requested subimage from that camera
      for (const hal_camera_image& subimage : param.subimage) {
        pv2.camera.map[pv2.camera.views.size()] = subimage.location;
        pv2.camera.views.push_back(subimage.location);
      }
    }

    /*
      Wrapper
    */

    if constexpr (Driver::Config::kInput == Driver::Input::HAL) {
      pv2.camera.wrapper = std::make_unique<CameraWrapper>();
      pv2.camera.wrapper->onInit(      //
          "front_low_res_15fps.yaml",  //
          pv2.camera.models,           //
          pv2.params.target_system);

      pv2.camera.wrapper->registerImageCallback(std::bind(
          on_pv2_capture_hal, std::ref(driver), std::placeholders::_1));
    }
  }

  //
  // CAN
  //

  {
    /*
      Wrapper
    */

    pv2.can.bus0 = std::make_unique<CanWrapper>("vehicle_can.yaml");
    pv2.can.bus1 = std::make_unique<CanWrapper>("vehicle_can.yaml");
    pv2.can.parser = std::make_unique<VehicleStateParser>();
  }

  //
  // Network
  //

  {
    /*
      Wrapper
    */

    if constexpr (Driver::Config::kInput == Driver::Input::Network) {
      pv2.network.wrapper = std::make_unique<NetRxWrapper>();
      pv2.network.wrapper->onInit("capture_sim.yaml");
      pv2.network.wrapper->registerRxCallback(std::bind(
          on_pv2_capture_net, std::ref(driver), std::placeholders::_1));
    }
  }

  //
  // Message Queue
  //

  {
    for (uint32_t id = 0U; id < NUM_VISION_MESSAGES; ++id) {
      MessageQueue& queue = driver.pv2.messages[id];
      queue.SetMessageID(static_cast<VisionMessageID>(id));
      queue.SetQueueSize(pv2.params.main_task.message_queue_sizes[id]);
      queue.Reset();
    }
  }

  //
  // Camera Motion Task
  //

  if (pv2.params.camera_motion_task.enable_task) {
    using Task = CameraMotionTask;

    std::unique_ptr<Task> task = create_task<Task>(pv2, /* frequency = */ 0);
    task->InitializeTask(           //
        pv2.params.cameras_active,  //
        pv2.params.camera_motion_task);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Feature Track Task
  //

  if (pv2.params.feature_track_task.enable_task) {
    using Task = FeatureTrackTask;

    std::unique_ptr<Task> task = create_task<Task>(pv2, /* frequency = */ 0);
    task->InitializeTask(pv2.params.feature_track_task);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Lane Track Task
  //

  if (pv2.params.lane_task.enable_task) {
    using Task = LaneTask;

    std::unique_ptr<Task> task = create_task<Task>(pv2, /* frequency = */ 0);
    task->InitializeTask(                //
        pv2.params.cameras_active,       //
        pv2.params.lane_task,            //
        pv2.params.lane_blob,            //
        pv2.params.lane_pitch,           //
        pv2.params.lane_tracker,         //
        pv2.params.lane_ramp,            //
        pv2.params.lane_road_elevation,  //
        pv2.params.lane_assignment);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Object Track Task
  //

  if (pv2.params.object_task.enable_task) {
    using Task = ObjectTask;

    std::unique_ptr<Task> task = create_task<Task>(pv2, /* frequency = */ 0);
    task->InitializeTask(              //
        pv2.params.cameras_active,     //
        pv2.params.cameras_remote,     //
        pv2.params.object_task,        //
        pv2.params.object_tracker,     //
        pv2.params.object_estimator,   //
        pv2.params.object_freespace,   //
        pv2.params.center_path,        //
        pv2.params.construction_zone,  //
        pv2.params.vehicle_setting);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Visualizer Task
  //

  if (pv2.params.visualizer_task.enable_task) {
    using Task = VisualizerTask;

    std::unique_ptr<Task> task = create_task<Task>(pv2, /* frequency = */ 0);
    task->InitializeTask(                   //
        pv2.params.visualizer_task,         //
        pv2.params.visualizer_window,       //
        pv2.params.visualizer_draw,         //
        pv2.params.camera_calibrator_task,  //
        pv2.params.target_system,           //
        pv2.params.vehicle_name);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  return Status::Success;
}

Status
start(Driver::PV2& pv2) {
  // Start all PV2 tasks!
  for (const std::unique_ptr<VisionTaskBase>& task : pv2.tasks) {
    task->StartTask();
    task->SetRun();
  }

  // Start PV2 streaming if enabled.
  if (pv2.camera.wrapper) {
    pv2.camera.wrapper->startStreaming();
  }

  return Status::Success;
}

Status
shutdown(Driver::PV2& pv2) {
  // PV2 relies on RAII.  Manual cleanup is not necessary.  I'm only forcing
  // a call to destructors here to make multiple initialize() and shutdown()
  // sequences just a little nicer so deinitalization actually happens during
  // shutdown() as one expects as opposed to when these variables are
  // re-assigned during the next call to initialize() which would be the point
  // where the previous pointee's destructors would be called.

  pv2.tasks.clear();
  pv2.network.wrapper.reset();
  pv2.can.parser.reset();
  pv2.can.bus1.reset();
  pv2.can.bus0.reset();
  pv2.camera.wrapper.reset();
  pv2.camera.models.reset();

  return Status::Success;
}

}  // namespace pv2

//
// Phantom Vision 3
//

namespace pv3 {

Status
initialize(Driver& driver) {
  const Driver::PV2& pv2 = driver.pv2;
  Driver::PV3& pv3 = driver.pv3;

  struct {
    // Note that all initialization-time objects whose lifetime must exceed that
    // of vision::Parameters must be stored below!  Failure to do so will result
    // in undefined behavior as a result of PV3 accessing reclaimed stack memory
    // locations that no longer belong to the intended objects!

    struct {
      struct {
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
    // For each detected HAL camera
    for (uint16_t camera = 0U; camera < pv2.camera.params.size(); ++camera) {
      const camera_params_hal& param = pv2.camera.params[camera];

      // Ensure the output format is supported!
      if (vision::core::image::Format::Unknown ==
          convert(param.output_format)) {
        return Status::Failure;
      }

      // For each requested subimage from that camera
      for (const hal_camera_image& subimage : param.subimage) {
        // Convert PV2 CameraIDs to their corresponding PV3 camera::Views.
        const vision::camera::View view = convert(subimage.location);

        // Store the view id for runtime use.
        pv3.camera.map[subimage.location] = pv3.camera.views.size();
        pv3.camera.views.push_back(view);

        // Convert HAL subimages to PV3 crops.
        params.camera.capture.crops.push_back({
            /* .input = */
            {
                /* .channel = */ camera,
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
        /* .frequency = */ 0U,
        /* .optimize = */ vision::Parameters::Optimize::Latency,
        /* .crops = */
        {
            /* .data = */
            params.camera.capture.crops.data(),

            /* .count = */
            cast::numeric<uint32_t>(params.camera.capture.crops.size()),
        },
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
      /* Object */
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

        pv3.network.classes[cls.name].push_back(cls.ids);
      }

      /* Segmentation */
      for (const TidlSegmentationConfig& config :
           tidl.models.front.segmentation_config) {
        struct {
          const std::string name = "segmentation";
          ClassificationList ids;
        } cls;

        for (const std::string& name : config.classes) {
          cls.ids.push_back(phantomnet_seg_name(name));
        }

        pv3.network.classes[cls.name].push_back(cls.ids);
      }

      /* Traffic */
      for (const TidlDecoderConfig& config :
           tidl.models.traffic.decoder_config) {
        struct {
          const std::string name = "traffic";
          ClassificationList ids;
        } cls;

        for (const std::string& name : config.classes) {
          cls.ids.push_back(phantomnet_object_name(name));
        }

        pv3.network.classes[cls.name].push_back(cls.ids);
      }
    }

    /*
      Decoders
    */

    {
      for (const std::vector<float>& ratio :
           tidl.models.front.bbox_aspect_ratios) {
        params.inference.decode.object.ratios.push_back({
            /* .x = */ ratio[0U],
            /* .y = */ ratio[1U],
        });
      }

      for (const TidlDecoderConfig& config : tidl.models.front.decoder_config) {
        Decode::Parameters::Object::Decoder decoder{
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

        params.inference.decode.object.decoders.push_back(decoder);
      }
    }

    /*
      PV3
    */

    params.init.inference.pipeline = {
        /* .optimize = */ vision::Parameters::Optimize::Latency,
        /* .infer = */
        {
            /* .models = */
            {
                /* .main = */
                {
                    tidl.models.front.network_filename.c_str(),
                    tidl.models.front.config_filename.c_str(),
                },
                /* .tsr = */
                {
                    tidl.models.traffic.network_filename.c_str(),
                    tidl.models.traffic.config_filename.c_str(),
                },
            },
        },
        /* .decode = */
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
    };
  }

  //
  // Image
  //

  {
    /*
      PV3
    */

    params.init.image = {
        /* .convert = */
        {
            /* .format = */ vision::core::image::Format::iBGR,
        },
    };
  }

  //
  // Output
  //

  {
    driver.pv3.output = {};

    driver.pv3.output.camera.callbacks.capture =
        std::bind(on_pv3_capture, std::ref(driver), std::placeholders::_1);

    driver.pv3.output.inference.callbacks.pipeline =
        std::bind(on_pv3_inference, std::ref(driver), std::placeholders::_1);

    driver.pv3.output.image.callbacks.convert =
        std::bind(on_pv3_convert, std::ref(driver), std::placeholders::_1);
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
      // All acccess to the queue must be synchronized!
      Driver::PV3::Queue::Input::Image& queue = driver.pv3.queue.input.image;

      /* Critical Section */ {
        std::lock_guard lock(queue.mutex);

        queue.nv12.clear();
        queue.storage.clear();
      } /* Critical Section */
    }

    // Vehicle

    {
      // All acccess to the queue must be synchronized!
      Driver::PV3::Queue::Input::Vehicle& queue =
          driver.pv3.queue.input.vehicle;

      /* Critical Section */ {
        std::lock_guard lock(queue.mutex);

        queue.wsm = WheelSpeedMeasurements();
        queue.esp = EspMeasurements();
        queue.sa = SteeringAngle();
        queue.gs = GearState();
        queue.yaw = {};
      } /* Critical Section */
    }

    /*
      Output
    */

    // Convert

    {
      // All acccess to the queue must be synchronized!
      Driver::PV3::Queue::Output::Convert& queue =
          driver.pv3.queue.output.convert;

      /* Critical Section */ {
        std::lock_guard lock(queue.mutex);

        queue.quit = false;
        queue.frame = 0U;
        queue.rgb.clear();

        // Launch the background worker thread.
        queue.worker = std::thread(pv3_convert_worker, std::ref(driver));
      } /* Critical Section */
    }

    // Decode

    {
      // All acccess to the queue must be synchronized!
      Driver::PV3::Queue::Output::Decode& queue =
          driver.pv3.queue.output.decode;

      /* Critical Section */ {
        std::lock_guard lock(queue.mutex);

        queue.quit = false;
        queue.frame = 0U;
        queue.data.status = Status::Failure;

        // Launch the background worker thread.
        queue.worker = std::thread(pv3_inference_worker, std::ref(driver));
      } /* Critical Section */
    }
  }

  pv3.context = {};
  return initialize(pv3.context, params.init);
}

Status
shutdown(Driver::PV3& pv3) {
  {
    // All acccess to the queue must be synchronized!
    Driver::PV3::Queue& queue = pv3.queue;

    /* Critical Section */ {
      std::lock_guard lock(queue.output.convert.mutex);
      queue.output.convert.quit = true;
    } /* Critical Section */

    /* Critical Section */ {
      std::lock_guard lock(queue.output.decode.mutex);
      queue.output.decode.quit = true;
    } /* Critical Section */

    // Notify the threads of our termination request.
    queue.output.convert.produced.notify_all();
    queue.output.decode.produced.notify_all();

    // Join the threads.
    queue.output.convert.worker.join();
    queue.output.decode.worker.join();
  }

  return vision::shutdown(pv3.context);
}

vision::camera::View
convert(const CameraID location) noexcept {
  using View = vision::camera::View;
  View view = View::Max;

  switch (location) {
    case CAM_FRONT_CENTER:
      view = View::Front_Center;
      break;

    case CAM_FRONT_CENTER_CROP:
      view = View::Front_Center_Crop;
      break;

    case CAM_FRONT_CENTER_MIDDLE:
      view = View::Front_Center_Middle;
      break;

    case CAM_FRONT_CENTER_WHOLE:
      view = View::Front_Center_Whole;
      break;

    case CAM_FRONT_CENTER_FULL:
      view = View::Front_Center_Full;
      break;

    case CAM_FRONT_CENTER_NARROW:
      view = View::Front_Center_Narrow;
      break;

    case CAM_FRONT_CENTER_NARROW_CROP:
      view = View::Front_Center_Narrow_Crop;
      break;

    case CAM_FRONT_CENTER_NARROW_FULL:
      view = View::Front_Center_Narrow_Full;
      break;

    case CAM_FRONT_CENTER_SVM:
      view = View::Front_Center_SVM;
      break;

    case CAM_FRONT_CENTER_SVM_CROP:
      view = View::Front_Center_SVM_Crop;
      break;

    case CAM_FRONT_CENTER_SVM_FULL:
      view = View::Front_Center_SVM_Full;
      break;

    case CAM_FRONT_LEFT:
      view = View::Front_Left;
      break;

    case CAM_FRONT_LEFT_CROP:
      view = View::Front_Left_Crop;
      break;

    case CAM_FRONT_RIGHT:
      view = View::Front_Right;
      break;

    case CAM_FRONT_RIGHT_CROP:
      view = View::Front_Right_Crop;
      break;

    case CAM_REAR_CENTER:
      view = View::Rear_Center;
      break;

    case CAM_REAR_CENTER_CROP:
      view = View::Rear_Center_Crop;
      break;

    case CAM_REAR_CENTER_FULL:
      view = View::Rear_Center_Full;
      break;

    case CAM_REAR_CENTER_SVM:
      view = View::Rear_Center_SVM;
      break;

    case CAM_REAR_CENTER_SVM_CROP:
      view = View::Rear_Center_SVM_Crop;
      break;

    case CAM_REAR_CENTER_SVM_FULL:
      view = View::Rear_Center_SVM_Full;
      break;

    case CAM_REAR_SIDE_LEFT:
      view = View::Rear_Left;
      break;

    case CAM_REAR_SIDE_LEFT_CROP:
      view = View::Rear_Left_Crop;
      break;

    case CAM_REAR_SIDE_LEFT_FULL:
      view = View::Rear_Left_Full;
      break;

    case CAM_REAR_SIDE_RIGHT:
      view = View::Rear_Right;
      break;

    case CAM_REAR_SIDE_RIGHT_CROP:
      view = View::Rear_Right_Crop;
      break;

    case CAM_REAR_SIDE_RIGHT_FULL:
      view = View::Rear_Right_Full;
      break;

    case CAM_SIDE_LEFT:
      view = View::Side_Left;
      break;

    case CAM_SIDE_LEFT_FULL:
      view = View::Side_Left_Full;
      break;

    case CAM_SIDE_RIGHT:
      view = View::Side_Right;
      break;

    case CAM_SIDE_RIGHT_FULL:
      view = View::Side_Right_Full;
      break;

    case CAM_SIDE_FRONT_LEFT:
      view = View::Side_Front_Left;
      break;

    case CAM_SIDE_FRONT_LEFT_FULL:
      view = View::Side_Front_Left_Full;
      break;

    case CAM_SIDE_FRONT_RIGHT:
      view = View::Side_Front_Right;
      break;

    case CAM_SIDE_FRONT_RIGHT_FULL:
      view = View::Side_Front_Right_Full;
      break;

    case CAM_SIDE_REAR_LEFT:
      view = View::Side_Rear_Left;
      break;

    case CAM_SIDE_REAR_LEFT_FULL:
      view = View::Side_Rear_Left_Full;
      break;

    case CAM_SIDE_REAR_RIGHT:
      view = View::Side_Rear_Right;
      break;

    case CAM_SIDE_REAR_RIGHT_FULL:
      view = View::Side_Rear_Right_Full;
      break;

    case CAM_GROUND:
      view = View::Ground;
      break;

    case CAM_FRONT_CENTER_CROP_BAYER:
    case CAM_RESERVED:
    case NUM_MAX_CAM_IDS:
    default:
      break;
  }

  PV3_ASSERT(View::Max != view, "Invalid!");
  return view;
}

vision::core::image::Format
convert(const hal_camera_format hal) noexcept {
  using Format = vision::core::image::Format;
  Format format = Format::Unknown;

  switch (hal) {
    case hal_camera_format::format_yuv420sp:
      format = Format::pNV12;
      break;

    case hal_camera_format::format_bgr:
      format = Format::pBGR;
      break;

    case hal_camera_format::format_rgb:
      format = Format::pRGB;

    case hal_camera_format::format_yuv422:
    default:
      break;
  }

  PV3_ASSERT(Format::Unknown != format, "Invalid!");
  return format;
}

// Note! This function assumes that the matrix outlives the image Handle.

vision::image::Handle
convert(const cv::Mat mat) noexcept {
  vision::image::Handle image{};

  switch (mat.depth()) {
    case CV_8UC1: {
      // Assuming CV_8UC1 == pNV12.  Not very robust!
      image.format = vision::core::image::Format::pNV12;
      image.width = cast::numeric<uint16_t>(mat.cols);
      image.stride = cast::numeric<uint16_t>(image.width * sizeof(uint8_t));
      image.height = cast::numeric<uint16_t>((mat.rows * 2U) / 3U);

      // Planes
      enum Plane : uint32_t {
        kLuma = 0U,
        kChroma = 1U,
      };

      // Luma
      image.planes[kLuma].as_void_ptr = mat.ptr();
      image.planes[kLuma].size = image.height * image.stride;

      // Chroma
      image.planes[kChroma].as_void_ptr = mat.ptr(image.height);
      image.planes[kChroma].size = image.planes[kLuma].size / 2U;
    } break;

    default:
      break;
  }

  PV3_ASSERT(image, "Invalid!");
  return image;
}

Decode::Parameters::Object::Type
convert(const TidlDecoderType type) noexcept {
  using Object = Decode::Parameters::Object;
  Object::Type dtype{Object::Type::Undefined};

  switch (type) {
    case TidlDecoderType::BBOX:
      dtype = Object::Type::Box;
      break;

    case TidlDecoderType::CUBOID:
      dtype = Object::Type::Cuboid;
      break;

    case TidlDecoderType::VRUBOX:
      dtype = Object::Type::VRU_AABB;
      break;

    case TidlDecoderType::VRUBOX_W_ORIENTATION:
      dtype = Object::Type::VRU_OBB;
      break;

    case TidlDecoderType::CLASSIFIER:
    case TidlDecoderType::MULTI_CLASSIFIER:
    case TidlDecoderType::UNKNOWN:
    default:
      break;
  }

  PV3_ASSERT(Object::Type::Undefined != dtype, "Invalid!");
  return dtype;
}

}  // namespace pv3

//
// Driver
//

void
handler(const int signal) noexcept {
  // Do not perform any other operations here in the signal handler except for
  // simply setting an atomic variable that signals a termination request! The
  // list of operations that may be safely performed inside a signal handler is
  // quite small.  Most operations that are reguarly safe are disallowed!
  terminate = signal;
}

Status
initialize(Driver& driver) noexcept {
  // Install signal handlers to properly shutdown vision on abrupt termination
  // requests.  It is important to properly shutdown subsystems and deinitialize
  // their corresponding hardware blocks on embedded hardware in the face of
  // unexpected termination requests.  Failure to do so may put the system in an
  // undesirable state requiring a restart.

  constexpr int kSignals[]{
      SIGINT,
      SIGQUIT,
      SIGTERM,
  };

  for (const int signal : kSignals) {
    errno = 0;

    if ((SIG_ERR == std::signal(signal, &handler)) || (0 != errno)) {
      return Status::Failure;
    }
  }

  try {
    // Initialize PV2
    Status status = pv2::initialize(driver);
    if (Status::Success != status) {
      return status;
    }

    // Initialize PV3
    status = pv3::initialize(driver);
    if (Status::Success != status) {
      return status;
    }

    // Start PV2
    status = pv2::start(driver.pv2);
    if (Status::Success != status) {
      return status;
    }
  }
  catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return Status::Failure;
  }
  catch (...) {
    std::cerr << "Exception: Unknown!" << std::endl;
    return Status::Failure;
  }

  return Status::Success;
}

Status
shutdown(Driver& driver) noexcept {
  try {
    // De-initialize PV3
    Status status = pv3::shutdown(driver.pv3);
    if (Status::Success != status) {
      return status;
    }

    // De-initialize PV2
    status = pv2::shutdown(driver.pv2);
    if (Status::Success != status) {
      return status;
    }
  }
  catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return Status::Failure;
  }
  catch (...) {
    std::cerr << "Exception: Unknown!" << std::endl;
    return Status::Failure;
  }

  return Status::Success;
}

}  // namespace
}  // namespace phantom_ai
