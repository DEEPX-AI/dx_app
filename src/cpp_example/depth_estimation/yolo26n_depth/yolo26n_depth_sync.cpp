/**
 * @file yolo26n_depth_sync.cpp
 * @brief Yolo26nDepthFactory synchronous inference example
 */

#include "factory/yolo26n_depth_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26nDepthFactory>();
    dxapp::SyncDepthRunner<dxapp::Yolo26nDepthFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
