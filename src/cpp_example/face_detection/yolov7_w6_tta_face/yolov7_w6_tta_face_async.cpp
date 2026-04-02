/**
 * @file yolov7_w6_tta_face_async.cpp
 * @brief YOLOv7_w6_tta_face asynchronous face detection example
 */

#include "factory/yolov7_w6_tta_face_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7_w6_tta_faceFactory>();
    dxapp::AsyncFaceRunner<dxapp::YOLOv7_w6_tta_faceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
