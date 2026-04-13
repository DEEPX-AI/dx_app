/**
 * @file vitb32_async.cpp
 * @brief Vitb32 asynchronous classification example
 */

#include "factory/vitb32_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Vitb32Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Vitb32Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
