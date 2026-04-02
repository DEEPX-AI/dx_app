/**
 * @file yolo26m_sync.cpp
 * @brief Yolo26m synchronous inference example
 */

#include "factory/yolo26m_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26mFactory>();
    dxapp::SyncDetectionRunner<dxapp::Yolo26mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
