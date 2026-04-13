/**
 * @file yolo26l_cls_async.cpp
 * @brief Yolo26l_cls asynchronous classification example
 */

#include "factory/yolo26l_cls_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26l_clsFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Yolo26l_clsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
