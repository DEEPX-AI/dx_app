/**
 * @file mediapipe_hand_detector_async.cpp
 * @brief MediapipeHandDetectorFactory asynchronous inference example
 */

#include "factory/mediapipe_hand_detector_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::MediapipeHandDetectorFactory>();
    dxapp::AsyncFaceRunner<dxapp::MediapipeHandDetectorFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
