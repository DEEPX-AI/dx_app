/**
 * @file resnext50_32x4d_hailo_async.cpp
 * @brief Resnext50_32x4d_hailo asynchronous classification example
 */

#include "factory/resnext50_32x4d_hailo_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resnext50_32x4d_hailoFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Resnext50_32x4d_hailoFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
