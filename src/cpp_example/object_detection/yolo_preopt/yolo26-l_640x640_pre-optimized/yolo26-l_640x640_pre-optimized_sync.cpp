/**
 * @file yolo26-l_640x640_pre-optimized
 * @brief yolo_preopt sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo26-l_640x640_pre-optimized_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo26_l_640x640_pre_optimized::YoloPreoptFactory>();
    dxapp::SyncDetectionRunner<dxapp::v_yolo26_l_640x640_pre_optimized::YoloPreoptFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
