/**
 * @file superpoint_480x640
 * @brief superpoint async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/superpoint_480x640_factory.hpp"
#include "common/runner/async_keypoint_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_superpoint_480x640::SuperpointFactory>();
    dxapp::AsyncKeypointRunner<dxapp::v_superpoint_480x640::SuperpointFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
