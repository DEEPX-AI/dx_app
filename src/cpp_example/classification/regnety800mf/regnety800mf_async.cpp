/**
 * @file regnety800mf_async.cpp
 * @brief Regnety800mf asynchronous classification example
 */

#include "factory/regnety800mf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety800mfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnety800mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
