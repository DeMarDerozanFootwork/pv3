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

namespace phantom_ai {
namespace driver {
namespace capture {

Status
initialize(Context& capture) {
  capture.backend = Backend::PV3;
  capture.params = capture::load(Config::Capture::kCamera);

  return Status::Success;
}

Status
start(driver::Context& driver) {
  if (Backend::PV3 != driver.capture.backend) {
    return Status::Failure;
  }

  return Status::Success;
}

Status
shutdown(Context& capture) {
  if (Backend::PV3 != capture.backend) {
    return Status::Failure;
  }

  return Status::Success;
}

}  // namespace capture
}  // namespace driver
}  // namespace phantom_ai
