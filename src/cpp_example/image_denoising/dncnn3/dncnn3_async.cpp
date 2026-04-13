/**
 * @file dncnn3_async.cpp
 * @brief DnCNN3 asynchronous image restoration example
 */

#include "factory/dncnn3_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN3Factory>();
    dxapp::AsyncRestorationRunner<dxapp::DnCNN3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
