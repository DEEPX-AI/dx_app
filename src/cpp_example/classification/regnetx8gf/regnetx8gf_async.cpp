/**
 * @file regnetx8gf_async.cpp
 * @brief Regnetx8gf asynchronous classification example
 */

#include "factory/regnetx8gf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx8gfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnetx8gfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
