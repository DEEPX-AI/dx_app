/**
 * @file yolo26x_obb_sync.cpp
 * @brief Yolo26x_obb synchronous inference example
 * 
 */

#include "factory/yolo26x_obb_factory.hpp"
#include "common/runner/sync_obb_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26x_obbFactory>();
    dxapp::SyncOBBRunner<dxapp::Yolo26x_obbFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
