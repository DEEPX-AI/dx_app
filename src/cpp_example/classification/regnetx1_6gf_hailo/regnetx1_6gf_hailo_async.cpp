/**
 * @file regnetx1_6gf_hailo_async.cpp
 * @brief Regnetx1_6gf_hailo asynchronous classification example
 */

#include "factory/regnetx1_6gf_hailo_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Regnetx1_6gf_hailoFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Regnetx1_6gf_hailoFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
