/**
 * @file yolo26-depth-n_768x768
 * @brief yolo26_depth async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo26-depth-n_768x768_factory.hpp"
#include "common/runner/async_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo26_depth_n_768x768::Yolo26DepthFactory>();
    dxapp::AsyncDepthRunner<dxapp::v_yolo26_depth_n_768x768::Yolo26DepthFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
