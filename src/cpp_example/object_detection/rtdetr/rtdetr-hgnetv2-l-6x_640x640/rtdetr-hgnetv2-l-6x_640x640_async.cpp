/**
 * @file rtdetr-hgnetv2-l-6x_640x640
 * @brief rtdetr async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/rtdetr-hgnetv2-l-6x_640x640_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_rtdetr_hgnetv2_l_6x_640x640::RtdetrFactory>();
    dxapp::AsyncDetectionRunner<dxapp::v_rtdetr_hgnetv2_l_6x_640x640::RtdetrFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
