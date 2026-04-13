/**
 * @file deepmar_resnet18_person_attr_resnet_v1_18_async.cpp
 * @brief Deepmar_ResNet18_person_attr_ResNet_v1_18 asynchronous classification example
 */

#include "factory/deepmar_resnet18_person_attr_resnet_v1_18_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deepmar_ResNet18_person_attr_ResNet_v1_18Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Deepmar_ResNet18_person_attr_ResNet_v1_18Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
