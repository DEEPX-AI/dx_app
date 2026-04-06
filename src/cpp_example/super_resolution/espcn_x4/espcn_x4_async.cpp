/**
 * @file espcn_x4_async.cpp
 * @brief ESPCN x4 asynchronous super-resolution example
 */

#include "factory/espcn_x4_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ESPCNX4Factory>();
    dxapp::AsyncRestorationRunner<dxapp::ESPCNX4Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
