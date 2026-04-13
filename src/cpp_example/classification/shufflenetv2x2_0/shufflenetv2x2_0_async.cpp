/**
 * @file shufflenetv2x2_0_async.cpp
 * @brief Shufflenetv2x2_0 asynchronous classification example
 */

#include "factory/shufflenetv2x2_0_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Shufflenetv2x2_0Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Shufflenetv2x2_0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
