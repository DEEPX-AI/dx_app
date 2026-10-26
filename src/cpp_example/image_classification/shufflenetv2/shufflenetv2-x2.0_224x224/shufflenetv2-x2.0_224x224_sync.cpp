/**
 * @file shufflenetv2-x2.0_224x224
 * @brief shufflenetv2 sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/shufflenetv2-x2.0_224x224_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_shufflenetv2_x2_0_224x224::Shufflenetv2Factory>();
    dxapp::SyncClassificationRunner<dxapp::v_shufflenetv2_x2_0_224x224::Shufflenetv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
