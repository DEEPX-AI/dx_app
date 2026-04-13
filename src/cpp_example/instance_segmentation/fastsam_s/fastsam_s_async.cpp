/**
 * @file fastsam_s_async.cpp
 * @brief YOLOv8Seg asynchronous instance segmentation example
 */

#include "factory/fastsam_s_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Fastsam_sFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::Fastsam_sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
