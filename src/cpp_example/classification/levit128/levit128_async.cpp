/**
 * @file levit128_async.cpp
 * @brief Levit128 asynchronous classification example
 */

#include "factory/levit128_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Levit128Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Levit128Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
