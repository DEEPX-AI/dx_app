/**
 * @file mnasnet1_0_async.cpp
 * @brief Mnasnet1_0 asynchronous classification example
 */

#include "factory/mnasnet1_0_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mnasnet1_0Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Mnasnet1_0Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
