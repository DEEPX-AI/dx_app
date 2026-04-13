/**
 * @file dncnn_gray_blind_async.cpp
 * @brief DnCNN_gray_blind asynchronous image restoration example
 */

#include "factory/dncnn_gray_blind_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DnCNN_gray_blindFactory>();
    dxapp::AsyncRestorationRunner<dxapp::DnCNN_gray_blindFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
