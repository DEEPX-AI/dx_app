/**
 * @file depthanythingv2-vits_224x224
 * @brief depthanythingv2 sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/depthanythingv2-vits_224x224_factory.hpp"
#include "common/runner/sync_depth_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_depthanythingv2_vits_224x224::Depthanythingv2Factory>();
    dxapp::SyncDepthRunner<dxapp::v_depthanythingv2_vits_224x224::Depthanythingv2Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
