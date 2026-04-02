#ifndef ESPCN_POSTPROCESS_H
#define ESPCN_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <string>
#include <vector>

struct ESPCNResult {
    std::vector<float> image{};
    int height{0};
    int width{0};
    int channels{1};

    ESPCNResult() = default;
    ESPCNResult(std::vector<float> img, int h, int w, int c = 1)
        : image(std::move(img)), height(h), width(w), channels(c) {}
    ~ESPCNResult() = default;
    ESPCNResult(const ESPCNResult&) = default;
    ESPCNResult& operator=(const ESPCNResult&) = default;
    ESPCNResult(ESPCNResult&&) noexcept = default;
    ESPCNResult& operator=(ESPCNResult&&) noexcept = default;
};

class ESPCNPostProcess {
   private:
    int input_width_{0};
    int input_height_{0};
    int scale_factor_{2};

   public:
    ESPCNPostProcess(int input_w, int input_h, int scale_factor = 2);
    ESPCNPostProcess();
    ~ESPCNPostProcess() = default;

    ESPCNResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    int get_scale_factor() const { return scale_factor_; }
};

#endif  // ESPCN_POSTPROCESS_H
