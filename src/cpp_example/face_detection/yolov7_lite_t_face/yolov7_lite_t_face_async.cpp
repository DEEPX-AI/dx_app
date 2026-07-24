/**
 * @file yolov7_lite_t_face_async.cpp
 * @brief Yolov7LiteTFaceFactory asynchronous inference example
 */

#include "factory/yolov7_lite_t_face_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov7LiteTFaceFactory>();
    dxapp::AsyncFaceRunner<dxapp::Yolov7LiteTFaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
