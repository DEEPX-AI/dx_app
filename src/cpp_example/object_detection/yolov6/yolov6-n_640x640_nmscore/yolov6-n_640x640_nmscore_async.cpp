/**
 * @file yolov6-n_640x640_nmscore
 * @brief yolov6 async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov6-n_640x640_nmscore_factory.hpp"
#include "common/runner/async_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov6_n_640x640_nmscore::Yolov6Factory>();
    dxapp::AsyncDetectionRunner<dxapp::v_yolov6_n_640x640_nmscore::Yolov6Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
