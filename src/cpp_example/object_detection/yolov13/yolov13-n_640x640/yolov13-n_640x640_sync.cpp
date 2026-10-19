/**
 * @file yolov13-n_640x640
 * @brief yolov13 sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov13-n_640x640_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov13_n_640x640::Yolov13Factory>();
    dxapp::SyncDetectionRunner<dxapp::v_yolov13_n_640x640::Yolov13Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
