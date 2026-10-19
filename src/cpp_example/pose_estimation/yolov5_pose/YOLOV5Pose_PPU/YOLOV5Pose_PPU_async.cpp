/**
 * @file YOLOV5Pose_PPU
 * @brief yolov5_pose async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/YOLOV5Pose_PPU_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_YOLOV5Pose_PPU::Yolov5PoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::v_YOLOV5Pose_PPU::Yolov5PoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
