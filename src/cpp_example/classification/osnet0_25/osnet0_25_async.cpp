/**
 * @file osnet0_25_async.cpp
 * @brief OSNet-0.25 asynchronous classification example
 */

#include "factory/osnet0_25_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Osnet025Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Osnet025Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
