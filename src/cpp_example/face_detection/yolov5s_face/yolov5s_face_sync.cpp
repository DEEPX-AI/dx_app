/**
 * @file yolov5s_face_sync.cpp
 * @brief YOLOv5s-Face synchronous face detection example
 */

#include "factory/yolov5s_face_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv5FaceFactory>();
    dxapp::SyncFaceRunner<dxapp::YOLOv5FaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
