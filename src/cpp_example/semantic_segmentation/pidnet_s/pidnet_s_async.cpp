/**
 * @file pidnet_s_async.cpp
 * @brief PidnetSFactory asynchronous inference example
 */

#include "factory/pidnet_s_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::PidnetSFactory>();
    dxapp::AsyncSemanticSegRunner<dxapp::PidnetSFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
