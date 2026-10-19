/**
 * @file yolo26-depth-x_768x768
 * @brief yolo26_depth sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo26-depth-x_768x768_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo26_depth_x_768x768::Yolo26DepthFactory>();
    dxapp::SyncDepthRunner<dxapp::v_yolo26_depth_x_768x768::Yolo26DepthFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
