/**
 * @file handlandmarklite_1_sync.cpp
 * @brief Hand Landmark Lite synchronous hand landmark example
 */

#include "factory/handlandmarklite_1_factory.hpp"
#include "common/runner/sync_hand_landmark_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Handlandmarklite_1Factory>();
    dxapp::SyncHandLandmarkRunner<dxapp::Handlandmarklite_1Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
