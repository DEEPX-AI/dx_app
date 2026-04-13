/**
 * @file yolo26m_seg_sync.cpp
 * @brief Yolo26m_seg synchronous instance segmentation example
 */

#include "factory/yolo26m_seg_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26m_segFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Yolo26m_segFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
