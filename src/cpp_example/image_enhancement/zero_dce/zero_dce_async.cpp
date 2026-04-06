/**
 * @file zero_dce_async.cpp
 * @brief Zero-DCE asynchronous image enhancement example
 */

#include "factory/zero_dce_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ZeroDCEFactory>();
    dxapp::AsyncRestorationRunner<dxapp::ZeroDCEFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
