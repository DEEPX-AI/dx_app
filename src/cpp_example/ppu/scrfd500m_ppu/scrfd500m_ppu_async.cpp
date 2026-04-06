/**
 * @file scrfd500m_ppu_async.cpp
 * @brief SCRFD500M-PPU asynchronous inference example
 */

#include "factory/scrfd500m_ppu_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SCRFD500m_ppuFactory>();
    dxapp::AsyncFaceRunner<dxapp::SCRFD500m_ppuFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
