/**
 * @file yolov5s_face_async.cpp
 * @brief YOLOv5Face asynchronous face detection example
 */

#include "factory/yolov5s_face_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5FaceFactory>();
    dxapp::AsyncFaceRunner<dxapp::YOLOv5FaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
