/**
 * @file yolov5m_face_sync.cpp
 * @brief YOLOv5m-Face synchronous face detection example
 */

#include "factory/yolov5m_face_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5m_faceFactory>();
    dxapp::SyncFaceRunner<dxapp::YOLOv5m_faceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
