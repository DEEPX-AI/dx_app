/**
 * @file realesrgan_x8_sync.cpp
 * @brief RealesrganX8Factory synchronous inference example
 */

#include "factory/realesrgan_x8_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RealesrganX8Factory>();
    dxapp::SyncRestorationRunner<dxapp::RealesrganX8Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
