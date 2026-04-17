/**
 * @file regnety200mf_async.cpp
 * @brief Regnety200mf asynchronous classification example
 */

#include "factory/regnety200mf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety200mfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnety200mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
