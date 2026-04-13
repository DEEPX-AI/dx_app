/**
 * @file shufflenetv2x1_5_async.cpp
 * @brief Shufflenetv2x1_5 asynchronous classification example
 */

#include "factory/shufflenetv2x1_5_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Shufflenetv2x1_5Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Shufflenetv2x1_5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
