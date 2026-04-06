/**
 * @file yolov7_w6_face_sync.cpp
 * @brief YOLOv7-W6-Face synchronous face detection example
 */

#include "factory/yolov7_w6_face_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7_w6_faceFactory>();
    dxapp::SyncFaceRunner<dxapp::YOLOv7_w6_faceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
