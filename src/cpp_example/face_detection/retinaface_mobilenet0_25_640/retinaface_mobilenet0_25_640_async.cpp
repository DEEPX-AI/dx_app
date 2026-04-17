/**
 * @file retinaface_mobilenet0_25_640_async.cpp
 * @brief RetinaFace asynchronous face detection example
 */

#include "factory/retinaface_mobilenet0_25_640_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::RetinaFaceFactory>();
    dxapp::AsyncFaceRunner<dxapp::RetinaFaceFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
