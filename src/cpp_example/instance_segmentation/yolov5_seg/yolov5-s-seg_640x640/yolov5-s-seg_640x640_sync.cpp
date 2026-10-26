/**
 * @file yolov5-s-seg_640x640
 * @brief yolov5_seg sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov5-s-seg_640x640_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov5_s_seg_640x640::Yolov5SegFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::v_yolov5_s_seg_640x640::Yolov5SegFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
