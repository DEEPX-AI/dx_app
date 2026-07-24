/**
 * @file levit192_async.cpp
 * @brief Levit192 asynchronous classification example
 */

#include "factory/levit192_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit192Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Levit192Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
