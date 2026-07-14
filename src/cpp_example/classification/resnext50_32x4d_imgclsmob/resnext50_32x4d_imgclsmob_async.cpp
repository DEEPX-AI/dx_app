/**
 * @file resnext50_32x4d_imgclsmob_async.cpp
 * @brief Resnext50_32x4d_imgclsmob asynchronous classification example
 */

#include "factory/resnext50_32x4d_imgclsmob_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Resnext50_32x4d_imgclsmobFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Resnext50_32x4d_imgclsmobFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
