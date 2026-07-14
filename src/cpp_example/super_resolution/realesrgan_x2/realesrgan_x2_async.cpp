/**
 * @file realesrgan_x2_async.cpp
 * @brief RealesrganX2Factory asynchronous inference example
 */

#include "factory/realesrgan_x2_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RealesrganX2Factory>();
    dxapp::AsyncRestorationRunner<dxapp::RealesrganX2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
