/**
 * @file pp-shituv1-mainbody-detection_640x640
 * @brief pp_shitu sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/pp-shituv1-mainbody-detection_640x640_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_pp_shituv1_mainbody_detection_640x640::PpShituFactory>();
    dxapp::SyncDetectionRunner<dxapp::v_pp_shituv1_mainbody_detection_640x640::PpShituFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
