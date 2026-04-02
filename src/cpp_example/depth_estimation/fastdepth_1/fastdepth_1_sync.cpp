/**
 * @file fastdepth_1_sync.cpp
 * @brief FastDepth_1 synchronous depth estimation example
 */

#include "factory/fastdepth_1_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::FastDepth_1Factory>();
    dxapp::SyncDepthRunner<dxapp::FastDepth_1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
