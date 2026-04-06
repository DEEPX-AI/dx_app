/**
 * @file osnet0_5_async.cpp
 * @brief OSNet-0.5 asynchronous classification example
 */

#include "factory/osnet0_5_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Osnet05Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Osnet05Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
