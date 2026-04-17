/**
 * @file wideresnet50_2_async.cpp
 * @brief WideResNet50_2 asynchronous classification example
 */

#include "factory/wideresnet50_2_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::WideResNet50_2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::WideResNet50_2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
