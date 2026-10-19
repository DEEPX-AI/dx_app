/**
 * @file yolov12-cls-l_224x224
 * @brief yolov12_cls async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov12-cls-l_224x224_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov12_cls_l_224x224::Yolov12ClsFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_yolov12_cls_l_224x224::Yolov12ClsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
