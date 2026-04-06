/**
 * @file hardnet68_async.cpp
 * @brief Hardnet68 asynchronous classification example
 */

#include "factory/hardnet68_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Hardnet68Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Hardnet68Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
