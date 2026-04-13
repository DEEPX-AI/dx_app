/**
 * @file densenet201_async.cpp
 * @brief Densenet201 asynchronous classification example
 */

#include "factory/densenet201_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Densenet201Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Densenet201Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
