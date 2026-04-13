/**
 * @file arcface_r50_async.cpp
 * @brief ArcFace MobileFaceNet asynchronous embedding example
 */

#include "factory/arcface_r50_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Arcface_r50Factory>();
    dxapp::AsyncEmbeddingRunner<dxapp::Arcface_r50Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}
