/**
 * @file arcface_mobilefacenet_async.cpp
 * @brief ArcFace MobileFaceNet asynchronous embedding example
 */

#include "factory/arcface_mobilefacenet_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::ArcFaceMobileFaceNetFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::ArcFaceMobileFaceNetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
