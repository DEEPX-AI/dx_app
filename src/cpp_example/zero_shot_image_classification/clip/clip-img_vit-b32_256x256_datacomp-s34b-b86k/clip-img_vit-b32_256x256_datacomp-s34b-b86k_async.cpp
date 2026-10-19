/**
 * @file clip-img_vit-b32_256x256_datacomp-s34b-b86k
 * @brief clip async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/clip-img_vit-b32_256x256_datacomp-s34b-b86k_factory.hpp"
#include "common/runner/async_embedding_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_clip_img_vit_b32_256x256_datacomp_s34b_b86k::ClipFactory>();
    dxapp::AsyncEmbeddingRunner<dxapp::v_clip_img_vit_b32_256x256_datacomp_s34b_b86k::ClipFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
