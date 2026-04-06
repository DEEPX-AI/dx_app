/**
 * @file dncnn_50_async.cpp
 * @brief DnCNN_50 asynchronous image restoration example
 */

#include "factory/dncnn_50_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN_50Factory>();
    dxapp::AsyncRestorationRunner<dxapp::DnCNN_50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
