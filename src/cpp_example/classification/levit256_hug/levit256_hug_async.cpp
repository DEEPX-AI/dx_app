/**
 * @file levit256_hug_async.cpp
 * @brief Levit256_hug asynchronous classification example
 */

#include "factory/levit256_hug_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit256_hugFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Levit256_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
