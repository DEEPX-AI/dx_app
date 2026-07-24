/**
 * @file levit384_async.cpp
 * @brief Levit384 asynchronous classification example
 */

#include "factory/levit384_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit384Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Levit384Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
