/**
 * @file face_attr_resnet_v1_18_async.cpp
 * @brief Face_attr_ResNet_v1_18 asynchronous classification example
 */

#include "factory/face_attr_resnet_v1_18_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Face_attr_ResNet_v1_18Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Face_attr_ResNet_v1_18Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
