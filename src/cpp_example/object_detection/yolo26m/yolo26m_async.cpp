/**
 * @file yolo26m_async.cpp
 * @brief Yolo26m asynchronous inference example
 */

#include "factory/yolo26m_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26mFactory>();
    dxapp::AsyncDetectionRunner<dxapp::Yolo26mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
