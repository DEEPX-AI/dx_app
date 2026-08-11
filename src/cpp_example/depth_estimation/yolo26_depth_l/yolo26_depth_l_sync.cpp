/**
 * @file yolo26_depth_l_sync.cpp
 * @brief Yolo26DepthLFactory synchronous inference example
 */

#include "factory/yolo26_depth_l_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthLFactory>();
    dxapp::SyncDepthRunner<dxapp::Yolo26DepthLFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
