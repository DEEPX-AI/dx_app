#ifndef CLIP_TEXT_POSTPROCESS_H
#define CLIP_TEXT_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <vector>

/**
 * @brief CLIP text encoder post-processing: the text embedding
 *
 * The encoder emits (1, D) or (1, seq_len, D). For the sequence form the
 * end-of-text position is taken: the arg-max of the token ids when a second
 * output carries them, otherwise the last position whose row is non-zero (CLIP
 * pads with zeros after EOS), falling back to the last position. The vector is
 * L2-normalised unless normalize is false. Same decode as the Python
 * CLIPTextPostprocessor.
 */
class CLIPTextPostProcess {
   private:
    bool normalize_{true};

   public:
    explicit CLIPTextPostProcess(bool normalize = true);
    ~CLIPTextPostProcess() = default;

    std::vector<float> postprocess(const dxrt::TensorPtrs& outputs);

    bool get_normalize() const { return normalize_; }
};

#endif  // CLIP_TEXT_POSTPROCESS_H
