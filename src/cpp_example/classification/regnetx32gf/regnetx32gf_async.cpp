/**
 * @file regnetx32gf_async.cpp
 * @brief Regnetx32gf asynchronous classification example
 */

#include "factory/regnetx32gf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx32gfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnetx32gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
