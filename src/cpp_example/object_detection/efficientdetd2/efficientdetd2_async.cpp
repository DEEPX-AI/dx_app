/**
 * @file efficientdetd2_async.cpp
 * @brief Efficientdetd2 asynchronous inference example
 */

#include "factory/efficientdetd2_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientdetd2Factory>();
    dxapp::AsyncDetectionRunner<dxapp::Efficientdetd2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
