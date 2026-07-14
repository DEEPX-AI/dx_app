/**
 * @file espcn_x2_async.cpp
 * @brief EspcnX2Factory asynchronous inference example
 */

#include "factory/espcn_x2_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::EspcnX2Factory>();
    dxapp::AsyncRestorationRunner<dxapp::EspcnX2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
