/**
 * @file scdepthv3_sync.cpp
 * @brief Scdepthv3 synchronous depth estimation example
 */

#include "factory/scdepthv3_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Scdepthv3Factory>();
    dxapp::SyncDepthRunner<dxapp::Scdepthv3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
