/**
 * @file mnasnet0_5_async.cpp
 * @brief Mnasnet0_5 asynchronous classification example
 */

#include "factory/mnasnet0_5_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mnasnet0_5Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Mnasnet0_5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
