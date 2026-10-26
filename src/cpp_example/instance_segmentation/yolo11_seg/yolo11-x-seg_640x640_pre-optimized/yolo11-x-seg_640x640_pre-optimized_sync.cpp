/**
 * @file yolo11-x-seg_640x640_pre-optimized
 * @brief yolo11_seg sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo11-x-seg_640x640_pre-optimized_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo11_x_seg_640x640_pre_optimized::Yolo11SegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::v_yolo11_x_seg_640x640_pre_optimized::Yolo11SegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
