/**
 * @file ulfgfd_rfb_320_async.cpp
 * @brief Ulfgfd_rfb_320 asynchronous face detection example
 */

#include "factory/ulfgfd_rfb_320_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Ulfgfd_rfb_320Factory>();
    dxapp::AsyncFaceRunner<dxapp::Ulfgfd_rfb_320Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
