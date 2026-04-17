/**
 * @file resnext50_32x4d_h_async.cpp
 * @brief Resnext50_32x4d_h asynchronous classification example
 */

#include "factory/resnext50_32x4d_h_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resnext50_32x4d_hFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Resnext50_32x4d_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
