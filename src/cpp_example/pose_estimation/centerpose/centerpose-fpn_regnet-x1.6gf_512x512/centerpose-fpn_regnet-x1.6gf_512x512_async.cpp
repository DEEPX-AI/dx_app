/**
 * @file centerpose-fpn_regnet-x1.6gf_512x512
 * @brief centerpose async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/centerpose-fpn_regnet-x1.6gf_512x512_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_centerpose_fpn_regnet_x1_6gf_512x512::CenterposeFactory>();
    dxapp::AsyncPoseRunner<dxapp::v_centerpose_fpn_regnet_x1_6gf_512x512::CenterposeFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
