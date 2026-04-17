/**
 * @file levit384_hug_async.cpp
 * @brief Levit384_hug asynchronous classification example
 */

#include "factory/levit384_hug_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit384_hugFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Levit384_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
