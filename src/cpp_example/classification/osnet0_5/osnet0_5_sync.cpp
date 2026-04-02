/**
 * @file osnet0_5_sync.cpp
 * @brief OSNet-0.5 synchronous classification example
 */

#include "factory/osnet0_5_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Osnet05Factory>();
    dxapp::SyncClassificationRunner<dxapp::Osnet05Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
