/**
 * @file espcn_x3_async.cpp
 * @brief ESPCN x4 asynchronous super-resolution example
 */

#include "factory/espcn_x3_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Espcn_x3Factory>();
    dxapp::AsyncRestorationRunner<dxapp::Espcn_x3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
