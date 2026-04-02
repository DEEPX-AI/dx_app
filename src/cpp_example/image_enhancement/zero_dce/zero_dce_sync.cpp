/**
 * @file zero_dce_sync.cpp
 * @brief Zero-DCE synchronous image enhancement example
 */

#include "factory/zero_dce_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ZeroDCEFactory>();
    dxapp::SyncRestorationRunner<dxapp::ZeroDCEFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
