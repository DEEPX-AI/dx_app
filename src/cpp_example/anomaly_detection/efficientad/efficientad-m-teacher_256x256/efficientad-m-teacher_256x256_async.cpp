/**
 * @file efficientad-m-teacher_256x256
 * @brief efficientad async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/efficientad-m-teacher_256x256_factory.hpp"
#include "common/runner/async_anomaly_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_efficientad_m_teacher_256x256::EfficientadFactory>();
    dxapp::AsyncAnomalyRunner<dxapp::v_efficientad_m_teacher_256x256::EfficientadFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
