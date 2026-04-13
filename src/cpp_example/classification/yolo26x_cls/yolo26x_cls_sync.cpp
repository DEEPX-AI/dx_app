/**
 * @file yolo26x_cls_sync.cpp
 * @brief Yolo26x_cls synchronous classification example using SyncClassificationRunner
 */

#include "factory/yolo26x_cls_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26x_clsFactory>();
    dxapp::SyncClassificationRunner<dxapp::Yolo26x_clsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
