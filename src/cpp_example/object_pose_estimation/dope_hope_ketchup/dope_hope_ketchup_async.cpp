/**
 * @file dope_hope_ketchup_async.cpp
 * @brief DopeHopeKetchupFactory asynchronous inference example
 */

#include "factory/dope_hope_ketchup_factory.hpp"
#include "common/runner/async_object_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DopeHopeKetchupFactory>();
    dxapp::AsyncObjectPoseRunner<dxapp::DopeHopeKetchupFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
