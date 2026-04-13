/**
 * @file resnext101_64x4d_async.cpp
 * @brief Resnext101_64x4d asynchronous classification example
 */

#include "factory/resnext101_64x4d_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resnext101_64x4dFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Resnext101_64x4dFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
