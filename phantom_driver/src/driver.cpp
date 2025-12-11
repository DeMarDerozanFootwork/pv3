/*******************************************************************************
 * @file    driver.cpp
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#include "driver.h"

namespace phantom_ai {
namespace driver {

Status
initialize(Context& driver) noexcept {
  try {
    // Initialize CAN
    Status status = can::initialize(driver.can);
    if (Status::Success != status) {
      return status;
    }

    // Initialize Capture
    status = capture::initialize(driver.capture);
    if (Status::Success != status) {
      return status;
    }

    // Initialize PV2
    status = pv2::initialize(driver);
    if (Status::Success != status) {
      return status;
    }

    // Initialize PV3
    status = pv3::initialize(driver);
    if (Status::Success != status) {
      return status;
    }

    // Start CAN
    status = can::start(driver);
    if (Status::Success != status) {
      return status;
    }

    // Start Capture
    status = capture::start(driver);
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
shutdown(Context& driver) noexcept {
  // Make sure to always let deinitialization run to completion despite
  // potential errors to give all subsystems and their corresponding HW blocks a
  // chance to shutdown.  Failure to do so may leak resources, or put the system
  // in an undesirable state, requiring a restart.

  Status status = Status::Success;

  try {
    // De-initialize PV3
    if (Status::Success != pv3::shutdown(driver)) {
      std::cerr << "pv3::shutdown() failed!" << std::endl;
      status = Status::Failure;
    }

    // De-initialize PV2
    if (Status::Success != pv2::shutdown(driver)) {
      std::cerr << "pv2::shutdown() failed!" << std::endl;
      status = Status::Failure;
    }

    // De-initialize Capture
    if (Status::Success != capture::shutdown(driver.capture)) {
      std::cerr << "capture::shutdown() failed!" << std::endl;
      status = Status::Failure;
    }

    // De-initialize CAN
    if (Status::Success != can::shutdown(driver.can)) {
      std::cerr << "can::shutdown() failed!" << std::endl;
      status = Status::Failure;
    }
  }
  catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    status = Status::Failure;
  }
  catch (...) {
    std::cerr << "Exception: Unknown!" << std::endl;
    status = Status::Failure;
  }

  return status;
}

}  // namespace driver
}  // namespace phantom_ai
