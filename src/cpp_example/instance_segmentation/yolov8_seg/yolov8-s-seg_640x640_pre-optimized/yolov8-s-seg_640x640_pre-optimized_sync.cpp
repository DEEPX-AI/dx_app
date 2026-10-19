/**
 * @file yolov8-s-seg_640x640_pre-optimized
 * @brief yolov8_seg sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov8-s-seg_640x640_pre-optimized_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov8_s_seg_640x640_pre_optimized::Yolov8SegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::v_yolov8_s_seg_640x640_pre_optimized::Yolov8SegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
