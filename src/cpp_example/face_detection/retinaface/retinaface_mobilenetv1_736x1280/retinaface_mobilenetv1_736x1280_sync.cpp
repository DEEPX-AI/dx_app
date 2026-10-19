/**
 * @file retinaface_mobilenetv1_736x1280
 * @brief retinaface sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/retinaface_mobilenetv1_736x1280_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_retinaface_mobilenetv1_736x1280::RetinafaceFactory>();
    dxapp::SyncFaceRunner<dxapp::v_retinaface_mobilenetv1_736x1280::RetinafaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
