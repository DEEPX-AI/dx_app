/**
 * @file scrfd2_5g_async.cpp
 * @brief SCRFD2_5g asynchronous face detection example
 */

#include "factory/scrfd2_5g_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SCRFD2_5gFactory>();
    dxapp::AsyncFaceRunner<dxapp::SCRFD2_5gFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
