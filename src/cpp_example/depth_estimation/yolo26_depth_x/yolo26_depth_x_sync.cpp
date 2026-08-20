/**
 * @file yolo26_depth_x_sync.cpp
 * @brief Yolo26DepthXFactory synchronous inference example
 */

#include "factory/yolo26_depth_x_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26DepthXFactory>();
    dxapp::SyncDepthRunner<dxapp::Yolo26DepthXFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
