/**
 * @file dncnn_50_sync.cpp
 * @brief DnCNN_50 synchronous image restoration example
 */

#include "factory/dncnn_50_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN_50Factory>();
    dxapp::SyncRestorationRunner<dxapp::DnCNN_50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
