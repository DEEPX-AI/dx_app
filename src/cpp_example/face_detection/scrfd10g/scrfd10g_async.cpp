/**
 * @file scrfd10g_async.cpp
 * @brief SCRFD10g asynchronous face detection example
 */

#include "factory/scrfd10g_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SCRFD10gFactory>();
    dxapp::AsyncFaceRunner<dxapp::SCRFD10gFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
