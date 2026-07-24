/**
 * @file retinaface_mobilenetv1_736_sync.cpp
 * @brief RetinafaceMobilenetv1736Factory synchronous inference example
 */

#include "factory/retinaface_mobilenet_v1_736x1280_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RetinafaceMobilenetv1736Factory>();
    dxapp::SyncFaceRunner<dxapp::RetinafaceMobilenetv1736Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
