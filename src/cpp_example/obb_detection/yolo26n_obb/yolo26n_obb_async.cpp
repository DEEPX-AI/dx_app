/**
 * @file yolo26n_obb_async.cpp
 * @brief Yolo26n_obb asynchronous inference example
 */

#include "factory/yolo26n_obb_factory.hpp"
#include "common/runner/async_obb_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26n_obbFactory>();
    dxapp::AsyncOBBRunner<dxapp::Yolo26n_obbFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}