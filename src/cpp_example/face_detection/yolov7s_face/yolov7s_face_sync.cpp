/**
 * @file yolov7s_face_sync.cpp
 * @brief YOLOv7s-Face synchronous face detection example
 */

#include "factory/yolov7s_face_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::YOLOv7s_faceFactory>();
    dxapp::SyncFaceRunner<dxapp::YOLOv7s_faceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
