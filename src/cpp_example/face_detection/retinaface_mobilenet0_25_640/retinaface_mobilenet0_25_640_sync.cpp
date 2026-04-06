/**
 * @file retinaface_mobilenet0_25_640_sync.cpp
 * @brief RetinaFace synchronous face detection example
 */

#include "factory/retinaface_mobilenet0_25_640_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RetinaFaceFactory>();
    dxapp::SyncFaceRunner<dxapp::RetinaFaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
