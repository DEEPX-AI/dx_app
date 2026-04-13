/**
 * @file yolo26s_cls_sync.cpp
 * @brief Yolo26s_cls synchronous classification example using SyncClassificationRunner
 */

#include "factory/yolo26s_cls_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26s_clsFactory>();
    dxapp::SyncClassificationRunner<dxapp::Yolo26s_clsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
