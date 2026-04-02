#ifndef EMBEDDING_POSTPROCESS_H
#define EMBEDDING_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <string>
#include <vector>

struct EmbeddingResult {
    std::vector<float> embedding{};
    int dimension{0};

    EmbeddingResult() = default;
    EmbeddingResult(std::vector<float> emb, int dim)
        : embedding(std::move(emb)), dimension(dim) {}
    ~EmbeddingResult() = default;
    EmbeddingResult(const EmbeddingResult&) = default;
    EmbeddingResult& operator=(const EmbeddingResult&) = default;
    EmbeddingResult(EmbeddingResult&&) noexcept = default;
    EmbeddingResult& operator=(EmbeddingResult&&) noexcept = default;
};

class EmbeddingPostProcess {
   private:
    bool l2_normalize_{true};

   public:
    explicit EmbeddingPostProcess(bool l2_normalize = true);
    ~EmbeddingPostProcess() = default;

    EmbeddingResult postprocess(const dxrt::TensorPtrs& outputs);

    bool get_l2_normalize() const { return l2_normalize_; }
};

#endif  // EMBEDDING_POSTPROCESS_H
