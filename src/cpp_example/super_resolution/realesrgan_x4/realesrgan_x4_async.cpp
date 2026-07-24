/**
 * @file realesrgan_x4_async.cpp
 * @brief RealesrganX4Factory asynchronous inference example
 */

#include "factory/realesrgan_x4_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RealesrganX4Factory>();
    dxapp::AsyncRestorationRunner<dxapp::RealesrganX4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
