/**
 * @file zero_dce_pp_sync.cpp
 * @brief ZeroDCEPPFactory synchronous inference example
 */

#include "factory/zero_dce_pp_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ZeroDCEPPFactory>();
    dxapp::SyncRestorationRunner<dxapp::ZeroDCEPPFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
