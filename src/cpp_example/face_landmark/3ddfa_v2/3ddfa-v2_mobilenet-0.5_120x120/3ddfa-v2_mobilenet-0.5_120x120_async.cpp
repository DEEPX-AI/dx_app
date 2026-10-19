/**
 * @file 3ddfa-v2_mobilenet-0.5_120x120
 * @brief 3ddfa_v2 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/3ddfa-v2_mobilenet-0.5_120x120_factory.hpp"
#include "common/runner/async_face_alignment_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_3ddfa_v2_mobilenet_0_5_120x120::N3ddfaV2Factory>();
    dxapp::AsyncFaceAlignmentRunner<dxapp::v_3ddfa_v2_mobilenet_0_5_120x120::N3ddfaV2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
