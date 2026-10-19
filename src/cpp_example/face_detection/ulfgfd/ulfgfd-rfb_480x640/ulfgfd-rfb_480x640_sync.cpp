/**
 * @file ulfgfd-rfb_480x640
 * @brief ulfgfd sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ulfgfd-rfb_480x640_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ulfgfd_rfb_480x640::UlfgfdFactory>();
    dxapp::SyncFaceRunner<dxapp::v_ulfgfd_rfb_480x640::UlfgfdFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
