/*******************************************************************************
 * @file    config.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_CONFIG_H
#define PHANTOM_DRIVER_SRC_CONFIG_H

namespace phantom_ai {
namespace driver {

// Configuration
struct Config final {
  struct CAN final {
    static constexpr const char kRxFile[]{"vehicle_can.yaml"};
    static constexpr const char kTxFile[]{"private_can.yaml"};

    static constexpr double kEpoch = 0.0;
  };

  struct Capture final {
    static constexpr const char kCamera[]{PHANTOM_DRIVER_CONFIG_CAPTURE_CAMERA};
    static constexpr const char kNetwork[]{"capture_sim.yaml"};
  };

  struct Vision final {
    static constexpr const char kFile[]{PHANTOM_DRIVER_CONFIG_VISION_FILE};
    static constexpr const char kRoot[]{PHANTOM_DRIVER_CONFIG_VISION_ROOT};
  };
};

}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_CONFIG_H */
