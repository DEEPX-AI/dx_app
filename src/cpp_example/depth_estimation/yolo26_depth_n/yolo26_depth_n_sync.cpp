/**
 * @file yolo26_depth_n_sync.cpp
 * @brief Yolo26DepthNFactory synchronous inference example
 */

#include "factory/yolo26_depth_n_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthNFactory>();
    dxapp::SyncDepthRunner<dxapp::Yolo26DepthNFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
