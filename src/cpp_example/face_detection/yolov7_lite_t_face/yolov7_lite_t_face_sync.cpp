/**
 * @file yolov7_lite_t_face_sync.cpp
 * @brief Yolov7LiteTFaceFactory synchronous inference example
 */

#include "factory/yolov7_lite_t_face_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolov7LiteTFaceFactory>();
    dxapp::SyncFaceRunner<dxapp::Yolov7LiteTFaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
