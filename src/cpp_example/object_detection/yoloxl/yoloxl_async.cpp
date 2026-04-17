/**
 * @file yoloxl_async.cpp
 * @brief YOLOXl asynchronous inference example
 */

#include "factory/yoloxl_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXlFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOXlFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
