/**
 * @file regnetx3_2gf_async.cpp
 * @brief Regnetx3_2gf asynchronous classification example
 */

#include "factory/regnetx3_2gf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx3_2gfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnetx3_2gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
