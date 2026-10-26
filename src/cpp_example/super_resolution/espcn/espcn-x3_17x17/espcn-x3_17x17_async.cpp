/**
 * @file espcn-x3_17x17
 * @brief espcn async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/espcn-x3_17x17_factory.hpp"
#include "common/runner/async_restoration_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_espcn_x3_17x17::EspcnFactory>();
    dxapp::AsyncRestorationRunner<dxapp::v_espcn_x3_17x17::EspcnFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
