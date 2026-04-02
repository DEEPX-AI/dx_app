/**
 * @file wideresnet101_2_async.cpp
 * @brief WideResNet101_2 asynchronous classification example
 */

#include "factory/wideresnet101_2_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::WideResNet101_2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::WideResNet101_2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
