/**
 * @file mnasnet0_75_async.cpp
 * @brief Mnasnet0_75 asynchronous classification example
 */

#include "factory/mnasnet0_75_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mnasnet0_75Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Mnasnet0_75Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
