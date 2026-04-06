/**
 * @file regnetx400mf_async.cpp
 * @brief Regnetx400mf asynchronous classification example
 */

#include "factory/regnetx400mf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx400mfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnetx400mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
