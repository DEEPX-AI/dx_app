/**
 * @file yolo26l_cls_sync.cpp
 * @brief Yolo26l_cls synchronous classification example using SyncClassificationRunner
 */

#include "factory/yolo26l_cls_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26l_clsFactory>();
    dxapp::SyncClassificationRunner<dxapp::Yolo26l_clsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
