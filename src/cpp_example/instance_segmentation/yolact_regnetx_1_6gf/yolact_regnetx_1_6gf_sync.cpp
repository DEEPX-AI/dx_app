/**
 * @file yolact_regnetx_1_6gf_sync.cpp
 * @brief YOLACT synchronous instance segmentation example
 */

#include "factory/yolact_regnetx_1_6gf_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolact_regnetx_1_6gfFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolact_regnetx_1_6gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
