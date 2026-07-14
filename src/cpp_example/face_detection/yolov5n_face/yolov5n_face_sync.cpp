/**
 * @file yolov5n_face_sync.cpp
 * @brief Yolov5nFaceFactory synchronous inference example
 */

#include "factory/yolov5n_face_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov5nFaceFactory>();
    dxapp::SyncFaceRunner<dxapp::Yolov5nFaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
