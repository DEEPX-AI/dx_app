/**
 * @file yolo11-n-pose_640x640_pre-optimized
 * @brief yolo11_pose async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo11-n-pose_640x640_pre-optimized_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo11_n_pose_640x640_pre_optimized::Yolo11PoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::v_yolo11_n_pose_640x640_pre_optimized::Yolo11PoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
