/**
 * @file ulfgfd_slim_320_without_postprocessing_async.cpp
 * @brief Ulfgfd_slim_320_without_postprocessing asynchronous face detection example
 */

#include "factory/ulfgfd_slim_320_without_postprocessing_factory.hpp"
#include "common/runner/async_face_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Ulfgfd_slim_320_without_postprocessingFactory>();
    dxapp::AsyncFaceRunner<dxapp::Ulfgfd_slim_320_without_postprocessingFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
