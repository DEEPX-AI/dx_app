/**
 * @file casvit-m_224x224
 * @brief casvit async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/casvit-m_224x224_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_casvit_m_224x224::CasvitFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::v_casvit_m_224x224::CasvitFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
