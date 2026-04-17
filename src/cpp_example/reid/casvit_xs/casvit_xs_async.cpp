/**
 * @file casvit_xs_async.cpp
 * @brief ArcFace MobileFaceNet asynchronous embedding example
 */

#include "factory/casvit_xs_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Casvit_xsFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::Casvit_xsFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
