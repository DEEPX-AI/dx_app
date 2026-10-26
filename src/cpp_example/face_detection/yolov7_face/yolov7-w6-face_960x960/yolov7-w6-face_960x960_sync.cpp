/**
 * @file yolov7-w6-face_960x960
 * @brief yolov7_face sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov7-w6-face_960x960_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov7_w6_face_960x960::Yolov7FaceFactory>();
    dxapp::SyncFaceRunner<dxapp::v_yolov7_w6_face_960x960::Yolov7FaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
