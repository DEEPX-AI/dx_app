/**
 * @file yolopv2_sync.cpp
 * @brief YOLOPv2Factory synchronous inference example
 */

#include "factory/yolopv2_factory.hpp"
#include "common/runner/sync_panoptic_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOPv2Factory>();
    dxapp::SyncPanopticRunner<dxapp::YOLOPv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
