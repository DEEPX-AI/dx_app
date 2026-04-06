/**
 * @file yolov7_face_sync.cpp
 * @brief YOLOv7-Face synchronous face detection example
 */

#include "factory/yolov7_face_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7_faceFactory>();
    dxapp::SyncFaceRunner<dxapp::YOLOv7_faceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
