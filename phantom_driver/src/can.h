/*******************************************************************************
 * @file    can.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_CAN_H
#define PHANTOM_DRIVER_SRC_CAN_H

#include "common.h"

namespace phantom_ai {
namespace driver {
namespace can {

struct Context final {
  std::unique_ptr<CanWrapper> bus0;
  std::unique_ptr<CanWrapper> bus1;
  std::unique_ptr<CanMessagePacker> packer;
  std::unique_ptr<VehicleStateParser> parser;
};

Status initialize(Context& can);
Status start(driver::Context& driver);
Status shutdown(Context& can);

}  // namespace can
}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_CAN_H */
