/**
 * @file clip-text_resnet50_77x512_openai
 * @brief clip async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/clip-text_resnet50_77x512_openai_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_clip_text_resnet50_77x512_openai::ClipFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::v_clip_text_resnet50_77x512_openai::ClipFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
