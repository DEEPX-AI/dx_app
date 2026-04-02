/**
 * @file yolo26n_obb_sync.cpp
 * @brief Yolo26n_obb synchronous inference example
 * 
 * Part of DX-APP v3.0.0 refactoring.
 */

#include "factory/yolo26n_obb_factory.hpp"
#include "common/runner/sync_obb_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26n_obbFactory>();
    dxapp::SyncOBBRunner<dxapp::Yolo26n_obbFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
