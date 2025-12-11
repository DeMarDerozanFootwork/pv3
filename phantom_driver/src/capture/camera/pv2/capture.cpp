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
    std::vector<CameraResults>& images) {
  PHANTOM_AI_PROFILE_FUNCTION;

  // PV3 (read-only reference)
  const pv3::Context& pv3 = driver.pv3;

  // All access to the queue must be synchronized!
  pv3::Context::Queue::Input::Image& queue = driver.pv3.queue.input.image;

  // Variable to track when to notify the consumer.  We only do so when new
  // images are available and only then if all received images are valid.
  uint32_t received = 0U;

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

        // cv::Mat CameraResults::image owns its memory.  The object's memory is
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

        // With the matrix kept alive, convert the cv::Mat to the equivalent PV3
        // image::Handles. Note that PV3 image::Handles are thin objects that
        // simply describe, and point to, a location in memory and do not own
        // the image data and its lifetime by design, hence, why we needed to
        // keep the matrix alive as the object in charge of the image memory.
        // This operation does not copy any substansive amount of data.
        queue.nv12[view] = utils::convert(result.image);

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

}  // namespace callbacks
}  // namespace

Status
initialize(Context& capture) {
  capture.backend = Backend::PV2;
  capture.params = capture::load(Config::Capture::kCamera);
  capture.impl.wrapper = std::make_unique<CameraWrapper>();

  return Status::Success;
}

Status
start(driver::Context& driver) {
  if ((Backend::PV2 != driver.capture.backend) ||
      !driver.capture.impl.wrapper) {
    return Status::Failure;
  }

  driver.capture.impl.wrapper->registerImageCallback(
      std::bind(callbacks::on_rx, std::ref(driver), std::placeholders::_1));

  driver.capture.impl.wrapper->onInit(  //
      Config::Capture::kCamera,         //
      driver.pv2.camera.models,         //
      driver.pv2.params.target_system);

  driver.capture.impl.wrapper->startStreaming();

  return Status::Success;
}

Status
shutdown(Context& capture) {
  if (Backend::PV2 != capture.backend) {
    return Status::Failure;
  }

  // This code relies on RAII.  Manual cleanup is not necessary.  I'm only
  // forcing a call to destructors here to make multiple initialize() and
  // shutdown() sequences just a little nicer so deinitalization actually
  // happens during shutdown() as one expects as opposed to when these variables
  // are re-assigned during the next call to initialize() which would be the
  // point where the previous pointee's destructors would have been called on.

  capture.impl.wrapper.reset();
  capture.params = {};

  return Status::Success;
}

}  // namespace capture
}  // namespace driver
}  // namespace phantom_ai
