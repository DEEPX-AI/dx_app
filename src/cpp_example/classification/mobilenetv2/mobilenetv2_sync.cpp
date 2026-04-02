/**
 * @file mobilenetv2_sync.cpp
 * @brief Mobilenetv2 synchronous classification example using SyncClassificationRunner
 */

#include "factory/mobilenetv2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mobilenetv2Factory>();
    dxapp::SyncClassificationRunner<dxapp::Mobilenetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
