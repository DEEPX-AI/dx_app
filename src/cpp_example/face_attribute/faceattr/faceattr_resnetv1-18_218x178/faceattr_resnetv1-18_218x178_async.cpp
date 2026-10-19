/**
 * @file faceattr_resnetv1-18_218x178
 * @brief faceattr async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/faceattr_resnetv1-18_218x178_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_faceattr_resnetv1_18_218x178::FaceattrFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_faceattr_resnetv1_18_218x178::FaceattrFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
