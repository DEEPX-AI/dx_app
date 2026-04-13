/**
 * @file fastsam_s_sync.cpp
 * @brief Fastsam_s synchronous instance segmentation example
 */

#include "factory/fastsam_s_factory.hpp"
#include "common/runner/sync_segmentation_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Fastsam_sFactory>();
    dxapp::SyncInstanceSegRunner<dxapp::Fastsam_sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
