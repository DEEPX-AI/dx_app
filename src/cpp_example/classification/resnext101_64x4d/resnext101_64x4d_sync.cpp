/**
 * @file resnext101_64x4d_sync.cpp
 * @brief Resnext101_64x4d synchronous classification example using SyncClassificationRunner
 */

#include "factory/resnext101_64x4d_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resnext101_64x4dFactory>();
    dxapp::SyncClassificationRunner<dxapp::Resnext101_64x4dFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
