/**
 * @file mediapipe_hand_detector_sync.cpp
 * @brief MediapipeHandDetectorFactory synchronous inference example
 */

#include "factory/mediapipe_hand_detector_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::MediapipeHandDetectorFactory>();
    dxapp::SyncFaceRunner<dxapp::MediapipeHandDetectorFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
