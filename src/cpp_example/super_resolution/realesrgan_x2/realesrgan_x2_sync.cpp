/**
 * @file realesrgan_x2_sync.cpp
 * @brief RealesrganX2Factory synchronous inference example
 */

#include "factory/realesrgan_x2_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RealesrganX2Factory>();
    dxapp::SyncRestorationRunner<dxapp::RealesrganX2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
