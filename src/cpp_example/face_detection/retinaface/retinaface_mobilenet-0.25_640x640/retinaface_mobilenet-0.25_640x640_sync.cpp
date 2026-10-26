/**
 * @file retinaface_mobilenet-0.25_640x640
 * @brief retinaface sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/retinaface_mobilenet-0.25_640x640_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_retinaface_mobilenet_0_25_640x640::RetinafaceFactory>();
    dxapp::SyncFaceRunner<dxapp::v_retinaface_mobilenet_0_25_640x640::RetinafaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
