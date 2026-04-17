/**
 * @file shufflenetv1x1_0_async.cpp
 * @brief Shufflenetv1x1_0 asynchronous classification example
 */

#include "factory/shufflenetv1x1_0_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Shufflenetv1x1_0Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Shufflenetv1x1_0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
