/**
 * @file yolov8-n_640x640
 * @brief yolov8 sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov8-n_640x640_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov8_n_640x640::Yolov8Factory>();
    dxapp::SyncDetectionRunner<dxapp::v_yolov8_n_640x640::Yolov8Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
