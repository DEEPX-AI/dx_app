/**
 * @file yolo26-s-pose_640x640_pre-optimized
 * @brief yolo_preopt_pose sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo26-s-pose_640x640_pre-optimized_factory.hpp"
#include "common/runner/sync_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo26_s_pose_640x640_pre_optimized::YoloPreoptPoseFactory>();
    dxapp::SyncPoseRunner<dxapp::v_yolo26_s_pose_640x640_pre_optimized::YoloPreoptPoseFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
