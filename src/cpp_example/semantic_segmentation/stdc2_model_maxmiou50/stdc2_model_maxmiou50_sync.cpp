/**
 * @file stdc2_seg50_sync.cpp
 * @brief Stdc2Seg50Factory synchronous inference example
 */

#include "factory/stdc2_model_maxmiou50_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Stdc2Seg50Factory>();
    dxapp::SyncSemanticSegRunner<dxapp::Stdc2Seg50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
