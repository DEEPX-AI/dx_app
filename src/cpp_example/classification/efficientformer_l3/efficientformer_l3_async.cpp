/**
 * @file efficientformer_l3_async.cpp
 * @brief Efficientformer_l3 asynchronous classification example
 */

#include "factory/efficientformer_l3_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientformer_l3Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Efficientformer_l3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
