/*******************************************************************************
 * @file    pv3.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_PV3_H
#define PHANTOM_DRIVER_SRC_PV3_H

#include "common.h"

namespace phantom_ai {
namespace driver {
namespace pv3 {

struct Context final {
  // PV3 context; a data structure used by PV3 to store persistent object state.
  // Refer to phantom_vision3/perception.h for detailed documentation.
  // Thread-safety: Not thread-safe! Opaque by design; contents to be ignored.
  vision::Context context;

  // PV3 per-frame output; may be ignored when data received through callbacks.
  // Refer to phantom_vision3/perception.h for detailed documentation.
  // Thread-safety: Not thread-safe!
  vision::Output output;

  // Backend-agnostic camera-related data used by PV3 tasks and callbacks.
  // Thread-safety: Read-write at init-time; strictly read-only at runtime.
  struct {
    std::unordered_map<CameraID, uint32_t /* PV3 View Index */> map;
    std::vector<vision::camera::View> views;
  } camera;

  // Neural Network helper data structures used for PV3 -> PV2 data conversions.
  // Thread-safety: Read-write at init-time; strictly read-only at runtime.
  struct {
    struct {
      std::unordered_map<std::string, std::vector<ClassificationList>> classes;
      alignas(16U) uint8_t map[16U];
    } front, traffic;

    struct {
      alignas(16U) uint8_t b[16U];
      alignas(16U) uint8_t g[16U];
      alignas(16U) uint8_t r[16U];
      alignas(16U) uint8_t a[16U];
      int32_t id;
    } visualizer;
  } network;

  // PV3 message queue; the communication hub between PV3 <-> {Driver|PV2}.
  struct Queue final {
    // PV3 Input Queue; any data to be sent to PV3 lives here.
    // Refer to phantom_vision3/perception.h for detailed documentation.
    struct Input final {
      // Input images; only used by {HAL|Network|TDA4x} capture; empty otherwise
      // Thread-safety: Not thread-safe! Access must be explicitly synchronized.
      struct Image final {
        std::mutex mutex;
        std::condition_variable produced;
        std::vector<cv::Mat> storage;
        std::vector<vision::image::Handle> nv12;
      } image;

      // Input vehicle state data; to be provided externally through CAN|rosbags
      // Thread-safety: Not thread-safe! Access must be explicitly synchronized.
      struct Vehicle final {
        std::mutex mutex;
        EspMeasurements esp;
        WheelSpeedMeasurements wsm;
        SteeringAngle sa;
        GearState gs;
        struct {
          float rate;
          float bias;
          float count;
        } yaw;
      } vehicle;
    } input;

    // PV3 Output Queue; any data to be received from PV3 lives here.
    // Refer to phantom_vision3/perception.h for detailed documentation.
    struct Output final {
      // The following list of PV3 output stages in use will gradually grow as
      // we retire PV2 tasks and bring the equivalent PV3 task online.

      // Camera output in grayscale / rgb format.
      // Refer to phantom_vision3/perception.h for detailed documentation.
      // Thread-safety: Not thread-safe! Access must be explicitly synchronized.
      struct Convert final {
        std::thread worker;
        std::mutex mutex;
        std::condition_variable produced;
        std::vector<vision::image::Handle> gray, rgb;
        uint32_t frame;
        bool quit;
      } convert;

      // Neural network output, post-processed and decoded, for downstream use.
      // Refer to phantom_vision3/perception.h for detailed documentation.
      // Thread-safety: Not thread-safe! Access must be explicitly synchronized.
      struct Decode final {
        std::thread worker;
        std::mutex mutex;
        std::condition_variable produced;
        vision::Output::Inference::Main::Decode data;
        uint32_t frame;
        bool quit;
      } decode;
    } output;
  } queue;
};

Status initialize(driver::Context& driver);
Status shutdown(driver::Context& driver);

}  // namespace pv3
}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_PV3_H */
