/**
 * @file ulfgfd_slim_320_without_postprocessing_sync.cpp
 * @brief Ulfgfd_slim_320_without_postprocessing synchronous face detection example
 */

#include "factory/ulfgfd_slim_320_without_postprocessing_factory.hpp"
#include "common/runner/sync_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Ulfgfd_slim_320_without_postprocessingFactory>();
    dxapp::SyncFaceRunner<dxapp::Ulfgfd_slim_320_without_postprocessingFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
