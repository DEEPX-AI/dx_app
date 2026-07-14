#ifndef DOPE_POSTPROCESS_H
#define DOPE_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <vector>

struct DopePeak {
    float x{0.0f};
    float y{0.0f};
    float confidence{0.0f};

    DopePeak() = default;
    DopePeak(float x_, float y_, float conf) : x(x_), y(y_), confidence(conf) {}
};

struct DopeResult {
    std::vector<DopePeak> peaks;  // 9 peaks: 8 vertices + 1 centroid

    DopeResult() = default;
    ~DopeResult() = default;
};

/**
 * @brief DOPE 6DoF object pose estimation postprocessor
 *
 * Input: [1, 25, H, W] — channels 0-8: belief maps, 9-24: affinity fields
 * Output: 9 belief map peaks (x, y, confidence) in model output coordinates.
 *         No scaling — caller handles coordinate transform to original image.
 */
class DOPEPostProcess {
   private:
    int input_width_{640};
    int input_height_{480};

    static constexpr int NUM_BELIEFS = 9;

   public:
    DOPEPostProcess(int input_w, int input_h);
    DOPEPostProcess();
    ~DOPEPostProcess() = default;

    DopeResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // DOPE_POSTPROCESS_H
