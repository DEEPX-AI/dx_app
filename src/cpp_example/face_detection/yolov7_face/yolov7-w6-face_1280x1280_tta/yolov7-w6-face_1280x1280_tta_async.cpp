/**
 * @file yolov7-w6-face_1280x1280_tta
 * @brief yolov7_face async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov7-w6-face_1280x1280_tta_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov7_w6_face_1280x1280_tta::Yolov7FaceFactory>();
    dxapp::AsyncFaceRunner<dxapp::v_yolov7_w6_face_1280x1280_tta::Yolov7FaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
