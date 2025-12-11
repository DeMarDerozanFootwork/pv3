/*******************************************************************************
 * @file    pv2.cpp
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#include "pv2.h"

#include "driver.h"

namespace phantom_ai {
namespace driver {
namespace pv2 {
namespace {

/*
  Utilities
*/

namespace utils {

template <typename Task>
std::unique_ptr<Task>
create(Context& pv2, const int frequency) {
  std::unique_ptr<Task> task = std::make_unique<Task>(pv2.messages);
  task->RegisterCameraModelList(pv2.camera.models);
  task->RegisterMainTaskParam(pv2.params.main_task);
  task->RegisterVisualizerTaskParam(pv2.params.visualizer_task);
  task->RegisterVisualizerWindowParam(pv2.params.visualizer_window);
  task->SetTimer(frequency);
  return task;
}

}  // namespace utils
}  // namespace

Status
initialize(driver::Context& driver) {
  // Alias
  pv2::Context& pv2 = driver.pv2;

  // Disable OpenCV's multithreading.
  cv::setNumThreads(0);

  //
  // Configuration
  //

  {
    const YamlNode root = load_yaml_file(  //
        Config::Vision::kRoot,             //
        Config::Vision::kFile);

    if (!root) {
      return Status::Failure;
    }

    if (!read_vision_params(root, pv2.params)) {
      return Status::Failure;
    }
  }

  //
  // Camera
  //

  {
    /*
      Models
    */

    pv2.camera.models = std::make_shared<CameraModelList>(NUM_MAX_CAM_IDS);
    pv2.camera.models->Initialize(               //
        pv2.params.camera_model_param_file,      //
        pv2.params.vehicle_name,                 //
        pv2.params.target_system,                //
        pv2.params.camera_image_size,            //
        pv2.params.cameras_active,               //
        pv2.params.cameras_image_mask,           //
        PhantomVisionDynamicCalibrationInput(),  //
        pv2.params.ros_node.verbosity & VERBOSITY_MAIN_CAMERA_CALIB);

    /*
      Views
    */

    // For each detected camera
    for (const auto& camera : driver.capture.params.camera) {
      // For each requested subimage from that camera
      for (const auto& subimage : camera.subimage) {
        // Construct PV2 <-> PV3 camera view mappings
        const CameraID id = subimage.location;
        pv2.camera.map[pv2.camera.views.size()] = id;
        pv2.camera.views.push_back(id);
      }
    }
  }

  //
  // Message Queue
  //

  {
    for (uint32_t id = 0U; id < pv2.messages.size(); ++id) {
      MessageQueue& queue = pv2.messages[id];
      queue.SetMessageID(static_cast<VisionMessageID>(id));
      queue.SetQueueSize(pv2.params.main_task.message_queue_sizes[id]);
      queue.Reset();
    }
  }

  //
  // AEB Task
  //

  if (pv2.params.aeb_task.enable_task) {
    using Task = AebTask;

    std::unique_ptr<Task> task = utils::create<Task>(  //
        pv2,                                           //
        /* frequency = */ 0);

    task->InitializeTask(           //
        pv2.params.cameras_active,  //
        pv2.params.aeb_task,        //
        pv2.params.vehicle_setting);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Camera Motion Task
  //

  if (pv2.params.camera_motion_task.enable_task) {
    using Task = CameraMotionTask;

    std::unique_ptr<Task> task = utils::create<Task>(  //
        pv2,                                           //
        /* frequency = */ 200);

    task->InitializeTask(           //
        pv2.params.cameras_active,  //
        pv2.params.camera_motion_task);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Feature Track Task
  //

  if (pv2.params.feature_track_task.enable_task) {
    using Task = FeatureTrackTask;

    std::unique_ptr<Task> task = utils::create<Task>(  //
        pv2,                                           //
        /* frequency = */ 200);

    task->InitializeTask(pv2.params.feature_track_task);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Lane Track Task
  //

  if (pv2.params.lane_task.enable_task) {
    using Task = LaneTask;

    std::unique_ptr<Task> task = utils::create<Task>(  //
        pv2,                                           //
        /* frequency = */ 200);

    task->InitializeTask(            //
        pv2.params.cameras_active,   //
        pv2.params.lane_task,        //
        pv2.params.lane_blob,        //
        pv2.params.lane_pitch,       //
        pv2.params.lane_tracker,     //
        pv2.params.lane_assignment,  //
        pv2.params.vehicle_setting);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Message Task
  //

  if (pv2.params.message_task.enable_task) {
    using Task = MessageTask;

    std::unique_ptr<Task> task = utils::create<Task>(  //
        pv2,                                           //
        /* frequency = */ pv2.params.message_task.task_hertz);

    task->InitializeTask(         //
        pv2.params.message_task,  //
        pv2.params.dynamic_calibration_task
            .enable_publish_dynamic_calibration_msg);

    task->RegisterCanPackCallback(
        [&driver](const PhantomVisionMeasurement& measurement) {
          driver.can.packer->receiveData(measurement);
        });

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Object Track Task
  //

  if (pv2.params.object_task.enable_task) {
    using Task = ObjectTask;

    std::unique_ptr<Task> task = utils::create<Task>(  //
        pv2,                                           //
        /* frequency = */ 200);

    task->InitializeTask(              //
        pv2.params.cameras_active,     //
        pv2.params.object_task,        //
        pv2.params.object_tracker,     //
        pv2.params.object_estimator,   //
        pv2.params.object_freespace,   //
        pv2.params.center_path,        //
        pv2.params.construction_zone,  //
        pv2.params.vehicle_setting);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  //
  // Visualizer Task
  //

  if (pv2.params.visualizer_task.enable_task) {
    using Task = VisualizerTask;

    std::unique_ptr<Task> task = utils::create<Task>(  //
        pv2,                                           //
        /* frequency = */ 200);

    task->InitializeTask(                   //
        pv2.params.visualizer_task,         //
        pv2.params.visualizer_window,       //
        pv2.params.visualizer_draw,         //
        pv2.params.camera_calibrator_task,  //
        pv2.params.target_system,           //
        pv2.params.vehicle_name);

    // Note that 'task' is moved from after this point and must not be used!
    pv2.tasks.push_back(std::move(task));
  }

  // Start all PV2 tasks!
  for (const std::unique_ptr<VisionTaskBase>& task : driver.pv2.tasks) {
    task->StartTask();
    task->SetRun();
  }

  return Status::Success;
}

Status
shutdown(driver::Context& driver) {
  // This code relies on RAII.  Manual cleanup is not necessary.  I'm only
  // forcing a call to destructors here to make multiple initialize() and
  // shutdown() sequences just a little nicer so deinitalization actually
  // happens during shutdown() as one expects as opposed to when these variables
  // are re-assigned during the next call to initialize() which would be the
  // point where the previous pointee's destructors would have been called on.

  driver.pv2.tasks.clear();
  driver.pv2.camera.models.reset();

  return Status::Success;
}

}  // namespace pv2
}  // namespace driver
}  // namespace phantom_ai
