/**
 * @file scrfd500m_sync.cpp
 * @brief SCRFD synchronous face detection example
 */

#include "factory/scrfd500m_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SCRFDFactory>();
    dxapp::SyncFaceRunner<dxapp::SCRFDFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
