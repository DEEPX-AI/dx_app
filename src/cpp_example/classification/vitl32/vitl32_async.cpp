/**
 * @file vitl32_async.cpp
 * @brief Vitl32 asynchronous classification example
 */

#include "factory/vitl32_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vitl32Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Vitl32Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
