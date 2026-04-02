#ifndef ZERO_DCE_POSTPROCESS_H
#define ZERO_DCE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <string>
#include <vector>

struct ZeroDCEResult {
    std::vector<float> image{};
    int height{0};
    int width{0};
    int channels{3};

    ZeroDCEResult() = default;
    ZeroDCEResult(std::vector<float> img, int h, int w, int c = 3)
        : image(std::move(img)), height(h), width(w), channels(c) {}
    ~ZeroDCEResult() = default;
    ZeroDCEResult(const ZeroDCEResult&) = default;
    ZeroDCEResult& operator=(const ZeroDCEResult&) = default;
    ZeroDCEResult(ZeroDCEResult&&) noexcept = default;
    ZeroDCEResult& operator=(ZeroDCEResult&&) noexcept = default;
};

class ZeroDCEPostProcess {
   private:
    int input_width_{256};
    int input_height_{256};

   public:
    ZeroDCEPostProcess(int input_w, int input_h);
    ZeroDCEPostProcess();
    ~ZeroDCEPostProcess() = default;

    ZeroDCEResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // ZERO_DCE_POSTPROCESS_H
