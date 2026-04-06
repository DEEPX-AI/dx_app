/**
 * @file densenet121_async.cpp
 * @brief Densenet121 asynchronous classification example
 */

#include "factory/densenet121_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Densenet121Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Densenet121Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
