/*******************************************************************************
 * @file    external.cpp
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#include "driver.h"

namespace phantom_ai {
namespace driver {

//
// IMPORTANT NOTE!
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
//

Status
process(Context& driver) {
  // Alias
  pv3::Context& pv3 = driver.pv3;

  /* Critical Section */ {
    // All access to the queue must be synchronized!
    pv3::Context::Queue::Input::Image& queue = pv3.queue.input.image;

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
                    cast::numeric<uint32_t>(pv3.camera.views.size()),
                },

                /* .images = */
                {
                    /* .data = */
                    queue.nv12.data(),

                    /* .count = */
                    cast::numeric<uint32_t>(queue.nv12.size()),
                },
            },
        },
    };

    // Call PV3!  Keep in mind that PV3 will actually not run in a critical
    // section for the entirety of the time vision::process() is executing as we
    // will unlock the mutex after PV3 is done with its first stage which would
    // be when the on_pv3_capture callback is called.
    return vision::process(pv3.context, input, pv3.output);
  } /* Critical Section */
}

}  // namespace driver
}  // namespace phantom_ai
