/**
 * @file casvit_s_async.cpp
 * @brief ArcFace MobileFaceNet asynchronous embedding example
 */

#include "factory/casvit_s_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Casvit_sFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::Casvit_sFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
