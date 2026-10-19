/**
 * @file yolo26-cls-x_224x224
 * @brief yolo26_cls async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolo26-cls-x_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolo26_cls_x_224x224::Yolo26ClsFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_yolo26_cls_x_224x224::Yolo26ClsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
