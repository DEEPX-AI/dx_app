/**
 * @file yolov6-n0_640x640_v0.2.1
 * @brief yolov6 sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov6-n0_640x640_v0.2.1_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov6_n0_640x640_v0_2_1::Yolov6Factory>();
    dxapp::SyncDetectionRunner<dxapp::v_yolov6_n0_640x640_v0_2_1::Yolov6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
