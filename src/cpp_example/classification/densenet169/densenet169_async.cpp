/**
 * @file densenet169_async.cpp
 * @brief Densenet169 asynchronous classification example
 */

#include "factory/densenet169_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Densenet169Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Densenet169Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
