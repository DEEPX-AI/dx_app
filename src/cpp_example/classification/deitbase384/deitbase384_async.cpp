/**
 * @file deitbase384_async.cpp
 * @brief Deitbase384 asynchronous classification example
 */

#include "factory/deitbase384_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Deitbase384Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Deitbase384Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
