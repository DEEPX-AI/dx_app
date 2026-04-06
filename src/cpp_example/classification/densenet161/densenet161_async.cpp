/**
 * @file densenet161_async.cpp
 * @brief Densenet161 asynchronous classification example
 */

#include "factory/densenet161_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Densenet161Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Densenet161Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
