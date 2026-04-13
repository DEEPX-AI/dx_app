/**
 * @file yolo26x_seg_sync.cpp
 * @brief Yolo26x_seg synchronous instance segmentation example
 */

#include "factory/yolo26x_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26x_segFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolo26x_segFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
