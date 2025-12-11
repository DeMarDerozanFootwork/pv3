/*******************************************************************************
 * @file    params.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_CAPTURE_CAMERA_PARAMS_HAL_IMPL_PARAMS_H
#define PHANTOM_DRIVER_SRC_CAPTURE_CAMERA_PARAMS_HAL_IMPL_PARAMS_H

#include "common.h"

// HAL
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include "hal_camera.h"
#pragma GCC diagnostic pop

namespace phantom_ai {
namespace driver {
namespace capture {

struct Parameters final {
  std::vector<camera_params> camera;
};

inline Parameters
load(const std::string& configuration) {
  Parameters params{};

  CameraInfoList info;
  std::vector<int32_t> annotation;

  readHalCamerasParameters(  //
      configuration,         //
      params.camera,         //
      info,                  //
      annotation,            //
      false,                 //
      false,                 //
      "tda4x");

  return params;
}

}  // namespace capture
}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_CAPTURE_CAMERA_PARAMS_HAL_IMPL_PARAMS_H */
