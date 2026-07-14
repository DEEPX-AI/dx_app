/**
 * @file efficientnetv2m_sync.cpp
 * @brief Efficientnetv2mFactory synchronous inference example
 */

#include "factory/efficientnetv2m_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientnetv2mFactory>();
    dxapp::SyncClassificationRunner<dxapp::Efficientnetv2mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
