#ifndef SUPERPOINT_POSTPROCESS_H
#define SUPERPOINT_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <vector>

struct SuperPointKeypoint {
    float x{0.0f};
    float y{0.0f};
    float score{0.0f};

    SuperPointKeypoint() = default;
    SuperPointKeypoint(float x_, float y_, float score_)
        : x(x_), y(y_), score(score_) {}
};

struct SuperPointResult {
    std::vector<SuperPointKeypoint> keypoints;
    std::vector<std::vector<float> > descriptors;

    SuperPointResult() = default;
    ~SuperPointResult() = default;
};

class SuperPointPostProcess {
   private:
    int input_width_{0};
    int input_height_{0};
    float conf_threshold_{0.015f};
    int top_k_{500};
    int nms_dist_{4};
    int border_remove_{4};

   public:
    SuperPointPostProcess(int input_w, int input_h,
                          float conf_threshold = 0.015f,
                          int top_k = 500,
                          int nms_dist = 4,
                          int border_remove = 4);
    SuperPointPostProcess();
    ~SuperPointPostProcess() = default;

    SuperPointResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_conf_threshold() const { return conf_threshold_; }
    int get_top_k() const { return top_k_; }
    void set_conf_threshold(float t) { conf_threshold_ = t; }
    void set_top_k(int k) { top_k_ = k; }
    void set_nms_dist(int d) { nms_dist_ = d; }
    void set_border_remove(int b) { border_remove_ = b; }
};

#endif  // SUPERPOINT_POSTPROCESS_H
