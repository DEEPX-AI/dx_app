/**
 * @file espcn_x2_sync.cpp
 * @brief EspcnX2Factory synchronous inference example
 */

#include "factory/espcn_x2_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EspcnX2Factory>();
    dxapp::SyncRestorationRunner<dxapp::EspcnX2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
