/**
 * @file yolov7_face_async.cpp
 * @brief YOLOv7_face asynchronous face detection example
 */

#include "factory/yolov7_face_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7_faceFactory>();
    dxapp::AsyncFaceRunner<dxapp::YOLOv7_faceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
