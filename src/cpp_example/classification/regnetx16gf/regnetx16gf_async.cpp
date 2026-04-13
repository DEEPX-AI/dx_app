/**
 * @file regnetx16gf_async.cpp
 * @brief Regnetx16gf asynchronous classification example
 */

#include "factory/regnetx16gf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx16gfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnetx16gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
