/**
 * @file sfa3d_608x608
 * @brief sfa3d sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/sfa3d_608x608_factory.hpp"
#include "common/runner/sync_3d_object_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_sfa3d_608x608::Sfa3dFactory>();
    dxapp::Sync3DDetectionRunner<dxapp::v_sfa3d_608x608::Sfa3dFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
