/*******************************************************************************
 * @file    driver.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_DRIVER_H
#define PHANTOM_DRIVER_SRC_DRIVER_H

#include "common.h"

#include "can.h"
#include "capture.h"
#include "pv2.h"
#include "pv3.h"

namespace phantom_ai {
namespace driver {

struct Context final {
  // CAN context; handles read and write interactions with the CAN bus.
  can::Context can;

  // Capture context; a data structure used by capture backends to store
  // persistent object state. Supported backends: {Network, PV2, or PV3}.
  capture::Context capture;

  // PV2 context; stores PV2-related persistent state handling PV2 interactions.
  pv2::Context pv2;

  // PV3 context; stores PV3-related persistent state handling PV3 interactions.
  pv3::Context pv3;
};

Status initialize(Context& driver) noexcept;
Status process(Context& driver);
Status shutdown(Context& driver) noexcept;

}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_DRIVER_H */
