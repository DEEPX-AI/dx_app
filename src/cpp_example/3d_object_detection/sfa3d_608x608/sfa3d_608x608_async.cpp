/**
 * @file sfa3d_608x608_async.cpp
 * @brief SFA3D asynchronous LiDAR 3D detection example
 */

#include "factory/sfa3d_608x608_factory.hpp"
#include "common/runner/async_3d_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::SFA3D608x608Factory>();
    dxapp::Async3DDetectionRunner<dxapp::SFA3D608x608Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
