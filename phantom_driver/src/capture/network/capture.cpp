/*******************************************************************************
 * @file    capture.cpp
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#include "capture.h"

#include "driver.h"
#include "utils.h"

namespace phantom_ai {
namespace driver {
namespace capture {
namespace {

/*
  Callbacks
*/

namespace callbacks {

void
on_rx(                        //
    driver::Context& driver,  //
    const Embedded_Frame& frame) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // PV3 (read-only reference)
  const pv3::Context& pv3 = driver.pv3;

  /*
    Capture
  */

  // Variable to track when to notify the consumer.  We only do so when new
  // images are available and only then if all received images are valid.
  uint32_t received = 0U;

  /* Critical Section */ {
    // All access to the queue must be synchronized!
    pv3::Context::Queue::Input::Image& queue = driver.pv3.queue.input.image;

    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // Pass 1:
    // We have the following two options in dealing with this queue: (1) either
    // block further production until all previously produced data is consumed,
    // or (2) drop data not yet consumed and replace it with a more recent
    // batch.  This implementation chooses the 2nd approach.  Note that there
    // will be  perceptible difference between the two options if consumption
    // keeps up with production at all times.  Otherwise, the first approach may
    // intorduce some jitter but will have the advantage of the system gathering
    // and processing more information from the environment.

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
            queue.nv12[view] = utils::convert(nv12);

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
    // All access to the queue must be synchronized!
    pv3::Context::Queue::Input::Vehicle& queue = driver.pv3.queue.input.vehicle;

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
                return cast::numeric<float>(vehicle.esp_count) /
                       cast::numeric<float>(vehicle.wheelspeed_count);
              }

              return 0.0F;
            }(),

            /* .wsm = */
            [&vehicle]() noexcept {
              if (vehicle.esp_count < vehicle.wheelspeed_count) {
                return 1.0F;
              }

              if (vehicle.esp_count > 0U) {
                return cast::numeric<float>(vehicle.wheelspeed_count) /
                       cast::numeric<float>(vehicle.esp_count);
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
              /* .esp = */ cast::numeric<uint32_t>(i * ratio.esp),
              /* .wsm = */ cast::numeric<uint32_t>(i * ratio.wsm),
          };

          if (index.esp < vehicle.esp_count) {
            convertEsp(vehicle.esp[index.esp], queue.esp);
          }

          if (index.wsm < vehicle.wheelspeed_count) {
            convertWheelSpeed(vehicle.wheelspeed[index.wsm], queue.wsm);
          }
        }

        utils::calculate_yaw_rate_bias(  //
            queue.esp,                   //
            queue.wsm,                   //
            queue.yaw.rate,              //
            queue.yaw.bias,              //
            queue.yaw.count);
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

}  // namespace callbacks
}  // namespace

Status
initialize(Context& capture) {
  capture.backend = Backend::Network;
  capture.params = capture::load(Config::Capture::kCamera);
  capture.impl.wrapper = std::make_unique<NetRxWrapper>();

  return Status::Success;
}

Status
start(driver::Context& driver) {
  if ((Backend::Network != driver.capture.backend) ||
      !driver.capture.impl.wrapper) {
    return Status::Failure;
  }

  driver.capture.impl.wrapper->registerRxCallback(
      std::bind(callbacks::on_rx, std::ref(driver), std::placeholders::_1));

  driver.capture.impl.wrapper->onInit(Config::Capture::kNetwork);

  return Status::Success;
}

Status
shutdown(Context& capture) {
  if (Backend::Network != capture.backend) {
    return Status::Failure;
  }

  // This code relies on RAII.  Manual cleanup is not necessary.  I'm only
  // forcing a call to destructors here to make multiple initialize() and
  // shutdown() sequences just a little nicer so deinitalization actually
  // happens during shutdown() as one expects as opposed to when these variables
  // are re-assigned during the next call to initialize() which would be the
  // point where the previous pointee's destructors would be called.

  capture.impl.wrapper.reset();
  capture.params = {};

  return Status::Success;
}

}  // namespace capture
}  // namespace driver
}  // namespace phantom_ai
