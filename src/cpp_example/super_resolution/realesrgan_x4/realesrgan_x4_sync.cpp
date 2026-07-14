/**
 * @file realesrgan_x4_sync.cpp
 * @brief RealesrganX4Factory synchronous inference example
 */

#include "factory/realesrgan_x4_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RealesrganX4Factory>();
    dxapp::SyncRestorationRunner<dxapp::RealesrganX4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
