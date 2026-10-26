/**
 * @file ulfgfd-rfb_240x320_nopp
 * @brief ulfgfd sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ulfgfd-rfb_240x320_nopp_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ulfgfd_rfb_240x320_nopp::UlfgfdFactory>();
    dxapp::SyncFaceRunner<dxapp::v_ulfgfd_rfb_240x320_nopp::UlfgfdFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
