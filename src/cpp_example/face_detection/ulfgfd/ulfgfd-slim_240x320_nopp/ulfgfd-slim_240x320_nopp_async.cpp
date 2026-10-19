/**
 * @file ulfgfd-slim_240x320_nopp
 * @brief ulfgfd async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ulfgfd-slim_240x320_nopp_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ulfgfd_slim_240x320_nopp::UlfgfdFactory>();
    dxapp::AsyncFaceRunner<dxapp::v_ulfgfd_slim_240x320_nopp::UlfgfdFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
