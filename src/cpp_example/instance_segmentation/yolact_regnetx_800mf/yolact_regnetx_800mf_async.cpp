/**
 * @file yolact_regnetx_800mf_async.cpp
 * @brief YOLACT asynchronous instance segmentation example
 */

#include "factory/yolact_regnetx_800mf_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLACTFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::YOLACTFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
