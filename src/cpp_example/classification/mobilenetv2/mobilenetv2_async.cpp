/**
 * @file mobilenetv2_async.cpp
 * @brief Mobilenetv2 asynchronous classification example
 */

#include "factory/mobilenetv2_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mobilenetv2Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Mobilenetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
