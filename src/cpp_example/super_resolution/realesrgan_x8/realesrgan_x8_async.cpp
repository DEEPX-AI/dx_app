/**
 * @file realesrgan_x8_async.cpp
 * @brief RealesrganX8Factory asynchronous inference example
 */

#include "factory/realesrgan_x8_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RealesrganX8Factory>();
    dxapp::AsyncRestorationRunner<dxapp::RealesrganX8Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
