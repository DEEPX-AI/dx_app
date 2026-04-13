/**
 * @file yoloxm_async.cpp
 * @brief YOLOXm asynchronous inference example
 */

#include "factory/yoloxm_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOXmFactory>();
    dxapp::AsyncDetectionRunner<dxapp::YOLOXmFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
