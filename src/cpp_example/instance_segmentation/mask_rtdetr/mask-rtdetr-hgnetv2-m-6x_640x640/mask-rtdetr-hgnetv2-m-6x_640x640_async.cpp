/**
 * @file mask-rtdetr-hgnetv2-m-6x_640x640
 * @brief mask_rtdetr async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/mask-rtdetr-hgnetv2-m-6x_640x640_factory.hpp"
#include "common/runner/async_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_mask_rtdetr_hgnetv2_m_6x_640x640::MaskRtdetrFactory>();
    dxapp::AsyncInstanceSegRunner<dxapp::v_mask_rtdetr_hgnetv2_m_6x_640x640::MaskRtdetrFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
