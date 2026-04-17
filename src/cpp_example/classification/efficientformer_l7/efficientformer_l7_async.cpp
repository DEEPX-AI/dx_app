/**
 * @file efficientformer_l7_async.cpp
 * @brief Efficientformer_l7 asynchronous classification example
 */

#include "factory/efficientformer_l7_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientformer_l7Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Efficientformer_l7Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
