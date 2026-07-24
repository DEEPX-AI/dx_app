/**
 * @file retinaface_mobilenetv1_736_async.cpp
 * @brief RetinafaceMobilenetv1736Factory asynchronous inference example
 */

#include "factory/retinaface_mobilenet_v1_736x1280_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RetinafaceMobilenetv1736Factory>();
    dxapp::AsyncFaceRunner<dxapp::RetinafaceMobilenetv1736Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
