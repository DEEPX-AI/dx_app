/**
 * @file nanodetplusm_async.cpp
 * @brief NanoDetplusm asynchronous inference example
 */

#include "factory/nanodetplusm_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDetplusmFactory>();
    dxapp::AsyncDetectionRunner<dxapp::NanoDetplusmFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
