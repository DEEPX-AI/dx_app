/**
 * @file dope_hope_ketchup_sync.cpp
 * @brief DopeHopeKetchupFactory synchronous inference example
 */

#include "factory/dope_hope_ketchup_factory.hpp"
#include "common/runner/sync_object_pose_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::DopeHopeKetchupFactory>();
    dxapp::SyncObjectPoseRunner<dxapp::DopeHopeKetchupFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
