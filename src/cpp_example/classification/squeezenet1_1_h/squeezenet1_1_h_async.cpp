/**
 * @file squeezenet1_1_h_async.cpp
 * @brief Squeezenet1_1_h asynchronous classification example
 */

#include "factory/squeezenet1_1_h_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Squeezenet1_1_hFactory>();
    dxapp::AsyncClassificationRunner<dxapp::Squeezenet1_1_hFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
