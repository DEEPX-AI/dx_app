/**
 * @file scrfd2_5g_sync.cpp
 * @brief SCRFD2_5g synchronous face detection example
 */

#include "factory/scrfd2_5g_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SCRFD2_5gFactory>();
    dxapp::SyncFaceRunner<dxapp::SCRFD2_5gFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
