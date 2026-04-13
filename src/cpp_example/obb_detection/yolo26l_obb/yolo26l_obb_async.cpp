/**
 * @file yolo26l_obb_async.cpp
 * @brief Yolo26l_obb asynchronous inference example
 * 
 */

#include "factory/yolo26l_obb_factory.hpp"
#include "common/runner/async_obb_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Yolo26l_obbFactory>();
    dxapp::AsyncOBBRunner<dxapp::Yolo26l_obbFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
