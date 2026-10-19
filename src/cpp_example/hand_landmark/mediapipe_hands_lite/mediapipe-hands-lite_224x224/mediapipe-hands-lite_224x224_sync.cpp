/**
 * @file mediapipe-hands-lite_224x224
 * @brief mediapipe_hands_lite sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/mediapipe-hands-lite_224x224_factory.hpp"
#include "common/runner/sync_hand_landmark_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_mediapipe_hands_lite_224x224::MediapipeHandsLiteFactory>();
    dxapp::SyncHandLandmarkRunner<dxapp::v_mediapipe_hands_lite_224x224::MediapipeHandsLiteFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
