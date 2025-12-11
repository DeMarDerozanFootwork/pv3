/*******************************************************************************
 * @file    can.cpp
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#include "can.h"

#include "driver.h"

namespace phantom_ai {
namespace driver {
namespace can {
namespace {

/*
  Callsbacks
*/

namespace callbacks {

void
on_can_esp_rx(                //
    driver::Context& driver,  //
    const EspMeasurements& esp) {
  /* Critical Section */ {
    // All access to the queue must be synchronized!
    pv3::Context::Queue::Input::Vehicle& queue = driver.pv3.queue.input.vehicle;

    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // Update value!
    queue.esp = esp;
  } /* Critical Section */
}

void
on_can_gs_rx(                 //
    driver::Context& driver,  //
    const GearState& gs) {
  /* Critical Section */ {
    // All access to the queue must be synchronized!
    pv3::Context::Queue::Input::Vehicle& queue = driver.pv3.queue.input.vehicle;

    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // Update value!
    queue.gs = gs;
  } /* Critical Section */
}

void
on_can_sa_rx(                 //
    driver::Context& driver,  //
    const SteeringAngle& sa) {
  /* Critical Section */ {
    // All access to the queue must be synchronized!
    pv3::Context::Queue::Input::Vehicle& queue = driver.pv3.queue.input.vehicle;

    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // Update value!
    queue.sa = sa;
  } /* Critical Section */
}

void
on_can_wsm_rx(                //
    driver::Context& driver,  //
    const WheelSpeedMeasurements& wsm) {
  /* Critical Section */ {
    // All access to the queue must be synchronized!
    pv3::Context::Queue::Input::Vehicle& queue = driver.pv3.queue.input.vehicle;

    // We cannot modify the queue if its contents are consumed in parallel.
    std::lock_guard<std::mutex> lock(queue.mutex);

    // Update value!
    queue.wsm = wsm;
  } /* Critical Section */
}

void
on_can_tx(                    //
    driver::Context& driver,  //
    const phantom_ai::CanFrameList& frames) {
  if (!driver.can.bus1->sendTxFrame(frames)) {
    // Log Error!
  }
}

}  // namespace callbacks

/*
  Utilities
*/

namespace utils {

std::vector<can_filter>
set_vehicle_message_filter() {
  std::vector<can_filter> filters;

  constexpr uint8_t kMessages = 4U;
  filters.reserve(kMessages);

  // Use single frame filter for all Rx messages on vehicle CAN.
  constexpr canid_t kMask = CAN_SFF_MASK;

  // Check DBC to select what we need to receive.
  filters.push_back({can_L_CGW_PC5_mid, kMask});
  filters.push_back({can_L_ESP12_mid, kMask});
  filters.push_back({can_L_SAS11_mid, kMask});
  filters.push_back({can_L_WHL_SPD11_mid, kMask});

  return filters;
}

}  // namespace utils
}  // namespace

Status
initialize(Context& can) {
  // CAN Bus 0
  std::vector<can_filter> filters = utils::set_vehicle_message_filter();
  can.bus0 = std::make_unique<CanWrapper>(Config::CAN::kRxFile);
  can.bus0->setMessageFilter(&filters, true);
  can.bus0->setTimestampOffset(Config::CAN::kEpoch);

  // CAN Bus 1
  can.bus1 = std::make_unique<CanWrapper>(Config::CAN::kTxFile);
  can.bus1->setTimestampOffset(Config::CAN::kEpoch);

  // Packer
  can.packer = std::make_unique<CanMessagePacker>();

  // Parser
  can.parser = std::make_unique<VehicleStateParser>();

  return Status::Success;
}

Status
start(driver::Context& driver) {
  /*
    Packer
  */

  driver.can.packer->init(
      std::bind(callbacks::on_can_tx, std::ref(driver), std::placeholders::_1),
      driver.pv2.params.message_task.enable_face_bounding_angle,
      (driver.pv2.params.message_task.verbosity & VERBOSITY_MESSAGE_CAN_PACKER)
          ? 3
          : 2,
      driver.pv2.params.message_task.can_max_object_counts);

  /*
    Parser
  */

  driver.can.parser->registerEspMeasurementsParserCallback(std::bind(
      callbacks::on_can_esp_rx, std::ref(driver), std::placeholders::_1));

  driver.can.parser->registerGearStateParserCallback(std::bind(
      callbacks::on_can_gs_rx, std::ref(driver), std::placeholders::_1));

  driver.can.parser->registerSteeringAngleParserCallback(std::bind(
      callbacks::on_can_sa_rx, std::ref(driver), std::placeholders::_1));

  driver.can.parser->registerWheelSpeedMeasurementsParserCallback(std::bind(
      callbacks::on_can_wsm_rx, std::ref(driver), std::placeholders::_1));

  driver.can.bus0->registerRxCallback([&driver](const CanFrame& frame) {
    driver.can.parser->parseVehicleStateMessage(frame);
  });

  return Status::Success;
}

Status
shutdown(Context& can) {
  // This code relies on RAII.  Manual cleanup is not necessary.  I'm only
  // forcing a call to destructors here to make multiple initialize() and
  // shutdown() sequences just a little nicer so deinitalization actually
  // happens during shutdown() as one expects as opposed to when these variables
  // are re-assigned during the next call to initialize() which would be the
  // point where the previous pointee's destructors would have been called on.

  can.parser.reset();
  can.packer.reset();
  can.bus1.reset();
  can.bus0.reset();

  return Status::Success;
}

}  // namespace can
}  // namespace driver
}  // namespace phantom_ai
