/**
 * @file scdepthv3_async.cpp
 * @brief Scdepthv3 asynchronous depth estimation example
 */

#include "factory/scdepthv3_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Scdepthv3Factory>();
    dxapp::AsyncDepthRunner<dxapp::Scdepthv3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
