/**
 * @file scrfd10g_sync.cpp
 * @brief SCRFD10g synchronous face detection example
 */

#include "factory/scrfd10g_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SCRFD10gFactory>();
    dxapp::SyncFaceRunner<dxapp::SCRFD10gFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
