/**
 * @file yolo26s_cls_async.cpp
 * @brief Yolo26s_cls asynchronous classification example
 */

#include "factory/yolo26s_cls_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26s_clsFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Yolo26s_clsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
