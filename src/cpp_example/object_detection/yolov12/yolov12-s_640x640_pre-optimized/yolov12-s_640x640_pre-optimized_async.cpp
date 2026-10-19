/**
 * @file yolov12-s_640x640_pre-optimized
 * @brief yolov12 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov12-s_640x640_pre-optimized_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov12_s_640x640_pre_optimized::Yolov12Factory>();
    dxapp::AsyncDetectionRunner<dxapp::v_yolov12_s_640x640_pre_optimized::Yolov12Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
