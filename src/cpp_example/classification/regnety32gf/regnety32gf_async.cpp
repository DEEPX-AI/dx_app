/**
 * @file regnety32gf_async.cpp
 * @brief Regnety32gf asynchronous classification example
 */

#include "factory/regnety32gf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety32gfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnety32gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
