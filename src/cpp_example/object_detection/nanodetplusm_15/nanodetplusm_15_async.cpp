/**
 * @file nanodetplusm_15_async.cpp
 * @brief NanoDetplusm_15 asynchronous inference example
 */

#include "factory/nanodetplusm_15_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::NanoDetplusm_15Factory>();
    dxapp::AsyncDetectionRunner<dxapp::NanoDetplusm_15Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
