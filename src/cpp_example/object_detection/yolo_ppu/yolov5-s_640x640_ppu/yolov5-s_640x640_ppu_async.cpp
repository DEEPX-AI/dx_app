/**
 * @file yolov5-s_640x640_ppu
 * @brief yolo_ppu async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov5-s_640x640_ppu_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov5_s_640x640_ppu::YoloPpuFactory>();
    dxapp::AsyncDetectionRunner<dxapp::v_yolov5_s_640x640_ppu::YoloPpuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
