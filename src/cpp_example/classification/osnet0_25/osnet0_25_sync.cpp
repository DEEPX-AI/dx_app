/**
 * @file osnet0_25_sync.cpp
 * @brief OSNet-0.25 synchronous classification example
 */

#include "factory/osnet0_25_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Osnet025Factory>();
    dxapp::SyncClassificationRunner<dxapp::Osnet025Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
