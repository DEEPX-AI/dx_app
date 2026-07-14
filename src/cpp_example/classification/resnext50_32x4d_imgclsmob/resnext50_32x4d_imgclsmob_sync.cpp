/**
 * @file resnext50_32x4d_imgclsmob_sync.cpp
 * @brief Resnext50_32x4d_imgclsmob synchronous classification example using SyncClassificationRunner
 */

#include "factory/resnext50_32x4d_imgclsmob_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resnext50_32x4d_imgclsmobFactory>();
    dxapp::SyncClassificationRunner<dxapp::Resnext50_32x4d_imgclsmobFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
