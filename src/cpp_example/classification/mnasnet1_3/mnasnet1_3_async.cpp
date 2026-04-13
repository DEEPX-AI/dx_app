/**
 * @file mnasnet1_3_async.cpp
 * @brief Mnasnet1_3 asynchronous classification example
 */

#include "factory/mnasnet1_3_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mnasnet1_3Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Mnasnet1_3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
