/**
 * @file inceptionv1_sync.cpp
 * @brief Inceptionv1 synchronous classification example using SyncClassificationRunner
 */

#include "factory/inceptionv1_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Inceptionv1Factory>();
    dxapp::SyncClassificationRunner<dxapp::Inceptionv1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
