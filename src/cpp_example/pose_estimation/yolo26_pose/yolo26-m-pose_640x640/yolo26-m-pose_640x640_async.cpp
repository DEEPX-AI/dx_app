/**
 * @file yolo26-m-pose_640x640
 * @brief yolo26_pose async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo26-m-pose_640x640_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo26_m_pose_640x640::Yolo26PoseFactory>();
    dxapp::AsyncPoseRunner<dxapp::v_yolo26_m_pose_640x640::Yolo26PoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
