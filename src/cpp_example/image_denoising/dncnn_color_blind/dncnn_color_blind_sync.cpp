/**
 * @file dncnn_color_blind_sync.cpp
 * @brief DnCNN_color_blind synchronous image restoration example
 */

#include "factory/dncnn_color_blind_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN_color_blindFactory>();
    dxapp::SyncRestorationRunner<dxapp::DnCNN_color_blindFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
