/**
 * @file regnety8gf_async.cpp
 * @brief Regnety8gf asynchronous classification example
 */

#include "factory/regnety8gf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety8gfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnety8gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
