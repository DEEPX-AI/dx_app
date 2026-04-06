/**
 * @file mobilenetv1_sync.cpp
 * @brief Mobilenetv1 synchronous classification example using SyncClassificationRunner
 */

#include "factory/mobilenetv1_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mobilenetv1Factory>();
    dxapp::SyncClassificationRunner<dxapp::Mobilenetv1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
