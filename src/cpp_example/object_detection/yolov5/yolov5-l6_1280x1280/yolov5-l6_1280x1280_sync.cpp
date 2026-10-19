/**
 * @file yolov5-l6_1280x1280
 * @brief yolov5 sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov5-l6_1280x1280_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov5_l6_1280x1280::Yolov5Factory>();
    dxapp::SyncDetectionRunner<dxapp::v_yolov5_l6_1280x1280::Yolov5Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
