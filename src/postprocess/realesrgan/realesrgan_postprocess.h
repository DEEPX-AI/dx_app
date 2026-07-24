#ifndef REALESRGAN_POSTPROCESS_H
#define REALESRGAN_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <vector>

struct RealESRGANResult {
    std::vector<float> image{};
    int height{0};
    int width{0};
    int channels{3};

    RealESRGANResult() = default;
    RealESRGANResult(std::vector<float> img, int h, int w, int c = 3)
        : image(std::move(img)), height(h), width(w), channels(c) {}
    ~RealESRGANResult() = default;
    RealESRGANResult(const RealESRGANResult&) = default;
    RealESRGANResult& operator=(const RealESRGANResult&) = default;
    RealESRGANResult(RealESRGANResult&&) noexcept = default;
    RealESRGANResult& operator=(RealESRGANResult&&) noexcept = default;
};

class RealESRGANPostProcess {
   private:
    int input_width_{0};
    int input_height_{0};
    int scale_factor_{4};

   public:
    RealESRGANPostProcess(int input_w, int input_h, int scale_factor = 4);
    RealESRGANPostProcess();
    ~RealESRGANPostProcess() = default;

    RealESRGANResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    int get_scale_factor() const { return scale_factor_; }
};

#endif  // REALESRGAN_POSTPROCESS_H
