/**
 * @file regnety16gf_async.cpp
 * @brief Regnety16gf asynchronous classification example
 */

#include "factory/regnety16gf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety16gfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnety16gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
