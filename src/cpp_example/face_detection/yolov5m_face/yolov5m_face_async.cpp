/**
 * @file yolov5m_face_async.cpp
 * @brief YOLOv5m_face asynchronous face detection example
 */

#include "factory/yolov5m_face_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_faceFactory>();
    dxapp::AsyncFaceRunner<dxapp::YOLOv5m_faceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
