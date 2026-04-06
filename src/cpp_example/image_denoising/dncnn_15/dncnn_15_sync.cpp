/**
 * @file dncnn_15_sync.cpp
 * @brief DnCNN_15 synchronous image restoration example
 */

#include "factory/dncnn_15_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN_15Factory>();
    dxapp::SyncRestorationRunner<dxapp::DnCNN_15Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
