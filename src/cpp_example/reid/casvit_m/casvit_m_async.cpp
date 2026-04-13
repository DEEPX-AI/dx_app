/**
 * @file casvit_m_async.cpp
 * @brief ArcFace MobileFaceNet asynchronous embedding example
 */

#include "factory/casvit_m_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Casvit_mFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::Casvit_mFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
