/**
 * @file fastvit-sa24_256x256
 * @brief fastvit async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/fastvit-sa24_256x256_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_fastvit_sa24_256x256::FastvitFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_fastvit_sa24_256x256::FastvitFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
