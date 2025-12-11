/*******************************************************************************
 * @file    capture.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_CAPTURE_H
#define PHANTOM_DRIVER_SRC_CAPTURE_H

#include "common.h"

#include "impl/capture.h"
#include "impl/params.h"

namespace phantom_ai {
namespace driver {
namespace capture {

enum class Backend : uint8_t {
  Network,
  PV2,
  PV3,
};

struct Context final {
  Backend backend;
  Parameters params;
  Impl impl;
};

Status initialize(Context& capture);
Status start(driver::Context& driver);
Status shutdown(Context& capture);

}  // namespace capture
}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_CAPTURE_H */
