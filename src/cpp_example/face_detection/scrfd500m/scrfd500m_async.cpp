/**
 * @file scrfd500m_async.cpp
 * @brief SCRFD asynchronous face detection example
 */

#include "factory/scrfd500m_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SCRFDFactory>();
    dxapp::AsyncFaceRunner<dxapp::SCRFDFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
