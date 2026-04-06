/**
 * @file casvit_t_async.cpp
 * @brief ArcFace MobileFaceNet asynchronous embedding example
 */

#include "factory/casvit_t_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Casvit_tFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::Casvit_tFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
