/**
 * @file centerpose_regnet-x800mf_640x640
 * @brief centerpose async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/centerpose_regnet-x800mf_640x640_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_centerpose_regnet_x800mf_640x640::CenterposeFactory>();
    dxapp::AsyncPoseRunner<dxapp::v_centerpose_regnet_x800mf_640x640::CenterposeFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
