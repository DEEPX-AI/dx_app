/**
 * @file regnety3_2gf_async.cpp
 * @brief Regnety3_2gf asynchronous classification example
 */

#include "factory/regnety3_2gf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnety3_2gfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnety3_2gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
