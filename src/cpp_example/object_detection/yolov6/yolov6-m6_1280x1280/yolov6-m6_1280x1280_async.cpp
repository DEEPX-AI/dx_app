/**
 * @file yolov6-m6_1280x1280
 * @brief yolov6 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov6-m6_1280x1280_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov6_m6_1280x1280::Yolov6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::v_yolov6_m6_1280x1280::Yolov6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
