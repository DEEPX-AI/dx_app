/**
 * @file scrfd-2.5g_640x640
 * @brief scrfd async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/scrfd-2.5g_640x640_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_scrfd_2_5g_640x640::ScrfdFactory>();
    dxapp::AsyncFaceRunner<dxapp::v_scrfd_2_5g_640x640::ScrfdFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
