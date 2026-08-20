/**
 * @file yolo26_depth_s_sync.cpp
 * @brief Yolo26DepthSFactory synchronous inference example
 */

#include "factory/yolo26_depth_s_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthSFactory>();
    dxapp::SyncDepthRunner<dxapp::Yolo26DepthSFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
