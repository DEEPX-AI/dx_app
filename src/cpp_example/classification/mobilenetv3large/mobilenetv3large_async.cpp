/**
 * @file mobilenetv3large_async.cpp
 * @brief Mobilenetv3large asynchronous classification example
 */

#include "factory/mobilenetv3large_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Mobilenetv3largeFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Mobilenetv3largeFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
