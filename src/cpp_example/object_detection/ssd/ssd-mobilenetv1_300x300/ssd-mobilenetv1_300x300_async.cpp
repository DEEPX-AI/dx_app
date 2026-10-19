/**
 * @file ssd-mobilenetv1_300x300
 * @brief ssd async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ssd-mobilenetv1_300x300_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ssd_mobilenetv1_300x300::SsdFactory>();
    dxapp::AsyncDetectionRunner<dxapp::v_ssd_mobilenetv1_300x300::SsdFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
