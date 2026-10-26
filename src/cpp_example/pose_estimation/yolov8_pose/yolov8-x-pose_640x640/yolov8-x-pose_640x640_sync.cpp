/**
 * @file yolov8-x-pose_640x640
 * @brief yolov8_pose sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov8-x-pose_640x640_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov8_x_pose_640x640::Yolov8PoseFactory>();
    dxapp::SyncPoseRunner<dxapp::v_yolov8_x_pose_640x640::Yolov8PoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
