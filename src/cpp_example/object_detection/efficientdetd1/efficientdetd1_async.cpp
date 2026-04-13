/**
 * @file efficientdetd1_async.cpp
 * @brief Efficientdetd1 asynchronous inference example
 */

#include "factory/efficientdetd1_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientdetd1Factory>();
    dxapp::AsyncDetectionRunner<dxapp::Efficientdetd1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
