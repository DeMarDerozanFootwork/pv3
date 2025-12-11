/*******************************************************************************
 * @file    capture.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_CAPTURE_CAMERA_PV2_IMPL_CAPTURE_H
#define PHANTOM_DRIVER_SRC_CAPTURE_CAMERA_PV2_IMPL_CAPTURE_H

#include "common.h"

namespace phantom_ai {
namespace driver {
namespace capture {

struct Impl final {
  std::unique_ptr<CameraWrapper> wrapper;
};

}  // namespace capture
}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_CAPTURE_CAMERA_PV2_IMPL_CAPTURE_H */
