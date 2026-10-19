/**
 * @file vitpose-s_256x192
 * @brief vitpose async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/vitpose-s_256x192_factory.hpp"
#include "common/runner/async_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_vitpose_s_256x192::VitposeFactory>();
    dxapp::AsyncPoseRunner<dxapp::v_vitpose_s_256x192::VitposeFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
