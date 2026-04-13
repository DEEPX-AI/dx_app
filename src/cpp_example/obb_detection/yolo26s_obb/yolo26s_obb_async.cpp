/**
 * @file yolo26s_obb_async.cpp
 * @brief Yolo26s_obb asynchronous inference example
 * 
 */

#include "factory/yolo26s_obb_factory.hpp"
#include "common/runner/async_obb_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26s_obbFactory>();
    dxapp::AsyncOBBRunner<dxapp::Yolo26s_obbFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
