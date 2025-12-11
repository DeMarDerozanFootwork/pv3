/*******************************************************************************
 * @file    pv2.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_PV2_H
#define PHANTOM_DRIVER_SRC_PV2_H

#include "common.h"

namespace phantom_ai {
namespace driver {
namespace pv2 {

struct Context final {
  // PV2 initialization-time configuration parameters.
  // Thread-safety: Read-write at init-time; strictly read-only at runtime.
  VisionParams params;

  // Backend-agnostic camera-related data used by PV2 tasks and callbacks.
  // Thread-safety: Read-write at init-time; strictly read-only at runtime.
  struct {
    CameraModelListS models;
    std::unordered_map<uint32_t /* PV3 View Index */, CameraID> map;
    std::vector<CameraID> views;
  } camera;

  // PV2 message queue; the internal communication hub between all PV2 tasks.
  // Thread-safety: Thread-safe!
  MessageContainer messages;

  // PV2 Vision Tasks
  // Thread-safety: Pointers strictly read-only; consult with objects otherwise.
  std::vector<std::unique_ptr<VisionTaskBase>> tasks;
};

Status initialize(driver::Context& driver);
Status shutdown(driver::Context& driver);

}  // namespace pv2
}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_PV2_H */
