/**
 * @file yolov9-t_640x640_pre-optimized
 * @brief yolov9 sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov9-t_640x640_pre-optimized_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov9_t_640x640_pre_optimized::Yolov9Factory>();
    dxapp::SyncDetectionRunner<dxapp::v_yolov9_t_640x640_pre_optimized::Yolov9Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
