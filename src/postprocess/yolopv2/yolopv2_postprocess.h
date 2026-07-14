#ifndef YOLOPV2_POSTPROCESS_H
#define YOLOPV2_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <array>
#include <vector>

struct YOLOPv2Detection {
    float x1{0.f}, y1{0.f}, x2{0.f}, y2{0.f};
    float confidence{0.f};
    int class_id{0};

    YOLOPv2Detection() = default;
    YOLOPv2Detection(float x1_, float y1_, float x2_, float y2_, float conf, int cls)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), confidence(conf), class_id(cls) {}
};

struct YOLOPv2Result {
    std::vector<YOLOPv2Detection> detections;
    std::vector<int> drivable_mask;  // [H*W]
    std::vector<int> lane_mask;      // [H*W]
    int mask_height{0};
    int mask_width{0};

    YOLOPv2Result() = default;
    ~YOLOPv2Result() = default;
};

class YOLOPv2PostProcess {
   private:
    int input_width_{640};
    int input_height_{384};
    float conf_threshold_{0.25f};
    float nms_threshold_{0.45f};

    static constexpr int NUM_CLASSES = 80;
    // YOLOPv2 only activates vehicle at index 3 (COCO class "car")
    static constexpr int VEHICLE_CLASS_INDEX = 3;
    static const float ANCHORS[3][3][2];
    static const int STRIDES[3];

    struct RawDet {
        float x1, y1, x2, y2, score;
    };

    std::vector<RawDet> decode_head(const float* data, int stride, int gh, int gw,
                                    const float anchors[][2]) const;
    std::vector<YOLOPv2Detection> nms_agnostic(std::vector<RawDet>& all_dets) const;

   public:
    YOLOPv2PostProcess(int input_w, int input_h, float conf_threshold = 0.25f,
                       float nms_threshold = 0.45f);
    YOLOPv2PostProcess();
    ~YOLOPv2PostProcess() = default;

    YOLOPv2Result postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_conf_threshold() const { return conf_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
};

#endif  // YOLOPV2_POSTPROCESS_H
