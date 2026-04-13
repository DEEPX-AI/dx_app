/**
 * @file 3ddfa_v2_mobilnet0_5_120x120_async.cpp
 * @brief 3DDFA v2 MobileNet 0.5 asynchronous face alignment example
 */

#include "factory/3ddfa_v2_mobilnet0_5_120x120_factory.hpp"
#include "common/runner/async_face_alignment_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Tddfa3ddfa_v2_mobilnet0_5_120x120Factory>();
    dxapp::AsyncFaceAlignmentRunner<dxapp::Tddfa3ddfa_v2_mobilnet0_5_120x120Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
