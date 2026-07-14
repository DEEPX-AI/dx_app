/**
 * @file yolov5n_face_async.cpp
 * @brief Yolov5nFaceFactory asynchronous inference example
 */

#include "factory/yolov5n_face_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov5nFaceFactory>();
    dxapp::AsyncFaceRunner<dxapp::Yolov5nFaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
