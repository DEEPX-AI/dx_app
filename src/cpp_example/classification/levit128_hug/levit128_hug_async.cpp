/**
 * @file levit128_hug_async.cpp
 * @brief Levit128_hug asynchronous classification example
 */

#include "factory/levit128_hug_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit128_hugFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Levit128_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
