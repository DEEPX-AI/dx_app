/**
 * @file mobilenetv1_async.cpp
 * @brief Mobilenetv1 asynchronous classification example
 */

#include "factory/mobilenetv1_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mobilenetv1Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Mobilenetv1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
