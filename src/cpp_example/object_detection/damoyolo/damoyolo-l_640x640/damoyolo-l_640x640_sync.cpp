/**
 * @file damoyolo-l_640x640
 * @brief damoyolo sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/damoyolo-l_640x640_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_damoyolo_l_640x640::DamoyoloFactory>();
    dxapp::SyncDetectionRunner<dxapp::v_damoyolo_l_640x640::DamoyoloFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
