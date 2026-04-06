/**
 * @file hardnet39ds_async.cpp
 * @brief Hardnet39ds asynchronous classification example
 */

#include "factory/hardnet39ds_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Hardnet39dsFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Hardnet39dsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
