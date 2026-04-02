/**
 * @file inceptionv1_async.cpp
 * @brief Inceptionv1 asynchronous classification example
 */

#include "factory/inceptionv1_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Inceptionv1Factory>();
    dxapp::AsyncClassificationRunner<dxapp::Inceptionv1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
