#ifndef MEDIAPIPE_HAND_POSTPROCESS_H
#define MEDIAPIPE_HAND_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <utility>
#include <vector>

struct MediaPipeHandDetection {
    float x1{0.f}, y1{0.f}, x2{0.f}, y2{0.f};
    float confidence{0.f};

    MediaPipeHandDetection() = default;
    MediaPipeHandDetection(float x1_, float y1_, float x2_, float y2_, float c)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), confidence(c) {}
};

class MediaPipeHandPostProcess {
   private:
    int input_size_{192};
    float score_threshold_{0.5f};
    float nms_threshold_{0.3f};
    float box_scale_{2.0f};
    float box_shift_{0.3f};

    std::vector<std::pair<float, float> > anchors_;

    void generate_anchors();

    struct RawDet {
        float cx;
        float cy;
        float s;
        float kx0;
        float ky0;
        float kx2;
        float ky2;
        float score;
    };

    std::vector<RawDet> decode_tensors(const dxrt::TensorPtrs& outputs) const;
    std::vector<MediaPipeHandDetection> apply_nms(std::vector<RawDet>& candidates) const;

   public:
    MediaPipeHandPostProcess(int input_size, float score_threshold, float nms_threshold);
    MediaPipeHandPostProcess();
    ~MediaPipeHandPostProcess() = default;

    std::vector<MediaPipeHandDetection> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_size_; }
    int get_input_height() const { return input_size_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
};

#endif  // MEDIAPIPE_HAND_POSTPROCESS_H
