/**
 * @file dncnn3_sync.cpp
 * @brief DnCNN3 synchronous image restoration example
 */

#include "factory/dncnn3_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN3Factory>();
    dxapp::SyncRestorationRunner<dxapp::DnCNN3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
