/**
 * @file resnext26-32x4d_224x224
 * @brief resnext sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/resnext26-32x4d_224x224_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_resnext26_32x4d_224x224::ResnextFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_resnext26_32x4d_224x224::ResnextFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
