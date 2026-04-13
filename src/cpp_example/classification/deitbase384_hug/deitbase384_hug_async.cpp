/**
 * @file deitbase384_hug_async.cpp
 * @brief Deitbase384_hug asynchronous classification example
 */

#include "factory/deitbase384_hug_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deitbase384_hugFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Deitbase384_hugFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
