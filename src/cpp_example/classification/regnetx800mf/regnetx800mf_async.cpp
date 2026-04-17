/**
 * @file regnetx800mf_async.cpp
 * @brief Regnetx800mf asynchronous classification example
 */

#include "factory/regnetx800mf_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx800mfFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnetx800mfFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
