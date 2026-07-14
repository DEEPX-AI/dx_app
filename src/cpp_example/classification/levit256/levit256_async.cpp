/**
 * @file levit256_async.cpp
 * @brief Levit256 asynchronous classification example
 */

#include "factory/levit256_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit256Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Levit256Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
