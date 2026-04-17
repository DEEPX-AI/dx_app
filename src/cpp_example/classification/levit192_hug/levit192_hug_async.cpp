/**
 * @file levit192_hug_async.cpp
 * @brief Levit192_hug asynchronous classification example
 */

#include "factory/levit192_hug_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit192_hugFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Levit192_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
