/**
 * @file yolact_regnetx_1_6gf_async.cpp
 * @brief YOLACT asynchronous instance segmentation example
 */

#include "factory/yolact_regnetx_1_6gf_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolact_regnetx_1_6gfFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Yolact_regnetx_1_6gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
