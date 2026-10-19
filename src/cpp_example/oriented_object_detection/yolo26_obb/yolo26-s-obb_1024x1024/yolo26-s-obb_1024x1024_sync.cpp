/**
 * @file yolo26-s-obb_1024x1024
 * @brief yolo26_obb sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo26-s-obb_1024x1024_factory.hpp"
#include "common/runner/sync_obb_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo26_s_obb_1024x1024::Yolo26ObbFactory>();
    dxapp::SyncOBBRunner<dxapp::v_yolo26_s_obb_1024x1024::Yolo26ObbFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
