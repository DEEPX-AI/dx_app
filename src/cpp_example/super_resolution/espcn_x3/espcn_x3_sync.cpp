/**
 * @file espcn_x3_sync.cpp
 * @brief ESPCN x4 synchronous super-resolution example
 */

#include "factory/espcn_x3_factory.hpp"
#include "common/runner/sync_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Espcn_x3Factory>();
    dxapp::SyncRestorationRunner<dxapp::Espcn_x3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
