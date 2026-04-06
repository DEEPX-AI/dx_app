/**
 * @file regnety400mf_async.cpp
 * @brief Regnety400mf asynchronous classification example
 */

#include "factory/regnety400mf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety400mfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnety400mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
