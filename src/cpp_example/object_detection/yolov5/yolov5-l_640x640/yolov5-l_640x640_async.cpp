/**
 * @file yolov5-l_640x640
 * @brief yolov5 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov5-l_640x640_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov5_l_640x640::Yolov5Factory>();
    dxapp::AsyncDetectionRunner<dxapp::v_yolov5_l_640x640::Yolov5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
