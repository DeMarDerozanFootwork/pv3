/*******************************************************************************
 * @file    common.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_COMMON_H
#define PHANTOM_DRIVER_SRC_COMMON_H

#include "fwd.h"

// Driver Configuration
#include "config.h"

// Phantom OS
#include <phantom_ai/core/profiler.h>
#include <phantom_ai/vehicle_state_parser/vehicle_state_parser.h>
#include <phantom_ai/vision_can_message_pack/can_message_pack.h>
#include <phantom_ai/wrappers/camera_wrapper.h>
#include <phantom_ai/wrappers/can_wrapper.h>
#include <phantom_ai/wrappers/network_rx_wrapper.h>

// Phantom Vision 2
#include <phantom_ai/phantom_vision2/aeb_task.h>
#include <phantom_ai/phantom_vision2/camera_motion_task.h>
#include <phantom_ai/phantom_vision2/feature_track_task.h>
#include <phantom_ai/phantom_vision2/lane_task.h>
#include <phantom_ai/phantom_vision2/message_task.h>
#include <phantom_ai/phantom_vision2/object_task.h>
#include <phantom_ai/phantom_vision2/visualizer_task.h>

// Phantom Vision 3
#include <phantom_ai/phantom_vision3/perception.h>

// CRT / STL
#include <csignal>

namespace phantom_ai {
namespace driver {

// Alias
namespace cast = vision::core::type::cast;

// Alias
using Status = vision::Status;

}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_COMMON_H */
