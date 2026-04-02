/**
 * @file yolact_regnetx_800mf_sync.cpp
 * @brief YOLACT synchronous instance segmentation example
 */

#include "factory/yolact_regnetx_800mf_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLACTFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::YOLACTFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
