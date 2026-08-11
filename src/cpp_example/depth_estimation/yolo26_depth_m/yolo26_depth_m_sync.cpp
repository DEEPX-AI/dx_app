/**
 * @file yolo26_depth_m_sync.cpp
 * @brief Yolo26DepthMFactory synchronous inference example
 */

#include "factory/yolo26_depth_m_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthMFactory>();
    dxapp::SyncDepthRunner<dxapp::Yolo26DepthMFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
