/**
 * @file ulfgfd_rfb_640_sync.cpp
 * @brief Ulfgfd_rfb_640 synchronous face detection example
 */

#include "factory/ulfgfd_rfb_640_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Ulfgfd_rfb_640Factory>();
    dxapp::SyncFaceRunner<dxapp::Ulfgfd_rfb_640Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
