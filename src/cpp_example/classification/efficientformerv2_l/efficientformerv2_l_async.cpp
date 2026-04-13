/**
 * @file efficientformerv2_l_async.cpp
 * @brief Efficientformerv2_l asynchronous classification example
 */

#include "factory/efficientformerv2_l_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Efficientformerv2_lFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Efficientformerv2_lFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
