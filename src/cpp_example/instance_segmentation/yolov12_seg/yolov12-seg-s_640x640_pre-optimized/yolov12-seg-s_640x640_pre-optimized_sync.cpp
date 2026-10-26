/**
 * @file yolov12-seg-s_640x640_pre-optimized
 * @brief yolov12_seg sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov12-seg-s_640x640_pre-optimized_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov12_seg_s_640x640_pre_optimized::Yolov12SegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::v_yolov12_seg_s_640x640_pre_optimized::Yolov12SegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
