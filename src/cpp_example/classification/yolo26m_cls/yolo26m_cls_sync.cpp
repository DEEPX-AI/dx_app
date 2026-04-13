/**
 * @file yolo26m_cls_sync.cpp
 * @brief Yolo26m_cls synchronous classification example using SyncClassificationRunner
 */

#include "factory/yolo26m_cls_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26m_clsFactory>();
    dxapp::SyncClassificationRunner<dxapp::Yolo26m_clsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
