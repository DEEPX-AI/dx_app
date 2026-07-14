/**
 * @file yolopv2_async.cpp
 * @brief YOLOPv2Factory asynchronous inference example
 */

#include "factory/yolopv2_factory.hpp"
#include "common/runner/async_panoptic_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOPv2Factory>();
    dxapp::AsyncPanopticRunner<dxapp::YOLOPv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
