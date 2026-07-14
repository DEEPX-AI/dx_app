/**
 * @file zero_dce_pp_async.cpp
 * @brief ZeroDCEPPFactory asynchronous inference example
 */

#include "factory/zero_dce_pp_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ZeroDCEPPFactory>();
    dxapp::AsyncRestorationRunner<dxapp::ZeroDCEPPFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
