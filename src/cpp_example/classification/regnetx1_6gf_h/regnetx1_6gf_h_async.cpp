/**
 * @file regnetx1_6gf_h_async.cpp
 * @brief Regnetx1_6gf_h asynchronous classification example
 */

#include "factory/regnetx1_6gf_h_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx1_6gf_hFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnetx1_6gf_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
