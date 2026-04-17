/**
 * @file dncnn_25_sync.cpp
 * @brief DnCNN_25 synchronous image restoration example
 */

#include "factory/dncnn_25_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN_25Factory>();
    dxapp::SyncRestorationRunner<dxapp::DnCNN_25Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
