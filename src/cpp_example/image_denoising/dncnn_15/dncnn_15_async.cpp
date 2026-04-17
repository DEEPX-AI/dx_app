/**
 * @file dncnn_15_async.cpp
 * @brief DnCNN_15 asynchronous image restoration example
 */

#include "factory/dncnn_15_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN_15Factory>();
    dxapp::AsyncRestorationRunner<dxapp::DnCNN_15Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
