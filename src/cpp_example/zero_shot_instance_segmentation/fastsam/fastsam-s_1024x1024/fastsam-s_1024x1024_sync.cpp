/**
 * @file fastsam-s_1024x1024
 * @brief fastsam sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/fastsam-s_1024x1024_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_fastsam_s_1024x1024::FastsamFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::v_fastsam_s_1024x1024::FastsamFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
