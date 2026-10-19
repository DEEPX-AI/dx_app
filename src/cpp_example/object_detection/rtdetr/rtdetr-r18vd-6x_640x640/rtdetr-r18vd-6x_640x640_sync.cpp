/**
 * @file rtdetr-r18vd-6x_640x640
 * @brief rtdetr sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/rtdetr-r18vd-6x_640x640_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_rtdetr_r18vd_6x_640x640::RtdetrFactory>();
    dxapp::SyncDetectionRunner<dxapp::v_rtdetr_r18vd_6x_640x640::RtdetrFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
