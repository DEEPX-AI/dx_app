#include "yolopv2_postprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

struct TensorView {
    const float* data;
    int channels;
    int height;
    int width;

    TensorView() : data(nullptr), channels(0), height(0), width(0) {}
};

TensorView make_tensor_view(const dxrt::TensorPtr& tensor) {
    const std::vector<int64_t>& shape = tensor->shape();
    if (shape.size() < 3) {
        throw std::runtime_error("YOLOPv2PostProcess: unexpected tensor rank");
    }

    TensorView view;
    view.data = static_cast<const float*>(tensor->data());
    view.channels = static_cast<int>(shape[shape.size() - 3]);
    view.height = static_cast<int>(shape[shape.size() - 2]);
    view.width = static_cast<int>(shape[shape.size() - 1]);
    return view;
}

float sigmoid(float x) {
    if (x < -30.0f) x = -30.0f;
    else if (x > 30.0f) x = 30.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

const float YOLOPv2PostProcess::ANCHORS[3][3][2] = {
    {{12.f, 16.f}, {19.f, 36.f}, {40.f, 28.f}},
    {{36.f, 75.f}, {76.f, 55.f}, {72.f, 146.f}},
    {{142.f, 110.f}, {192.f, 243.f}, {459.f, 401.f}},
};

const int YOLOPv2PostProcess::STRIDES[3] = {8, 16, 32};

YOLOPv2PostProcess::YOLOPv2PostProcess()
    : input_width_(640), input_height_(384), conf_threshold_(0.25f), nms_threshold_(0.45f) {}

YOLOPv2PostProcess::YOLOPv2PostProcess(int input_w, int input_h, float conf_threshold,
                                       float nms_threshold)
    : input_width_(input_w),
      input_height_(input_h),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold) {}

std::vector<YOLOPv2PostProcess::RawDet> YOLOPv2PostProcess::decode_head(
    const float* data, int stride, int gh, int gw, const float anchors[][2]) const {
    std::vector<RawDet> decoded;
    const int plane = gh * gw;
    const int fields = NUM_CLASSES + 5;

    for (int anchor = 0; anchor < 3; ++anchor) {
        for (int gy = 0; gy < gh; ++gy) {
            for (int gx = 0; gx < gw; ++gx) {
                const int offset = gy * gw + gx;
                const int base = anchor * fields;

                const float obj = sigmoid(data[(base + 4) * plane + offset]);
                // Vehicle class only (index 3), matching custom_ops.py vehicle_class_index
                const float cls_score =
                    sigmoid(data[(base + 5 + VEHICLE_CLASS_INDEX) * plane + offset]);
                const float score = obj * cls_score;
                if (score <= conf_threshold_) {
                    continue;
                }

                const float tx = data[(base + 0) * plane + offset];
                const float ty = data[(base + 1) * plane + offset];
                const float tw = data[(base + 2) * plane + offset];
                const float th = data[(base + 3) * plane + offset];

                const float cx = (sigmoid(tx) * 2.0f - 0.5f + static_cast<float>(gx)) * stride;
                const float cy = (sigmoid(ty) * 2.0f - 0.5f + static_cast<float>(gy)) * stride;
                const float bw = std::pow(sigmoid(tw) * 2.0f, 2.0f) * anchors[anchor][0];
                const float bh = std::pow(sigmoid(th) * 2.0f, 2.0f) * anchors[anchor][1];

                decoded.push_back(RawDet{cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f,
                                         cy + bh * 0.5f, score});
            }
        }
    }

    return decoded;
}

std::vector<YOLOPv2Detection> YOLOPv2PostProcess::nms_agnostic(
    std::vector<RawDet>& all_dets) const {
    std::sort(all_dets.begin(), all_dets.end(),
              [](const RawDet& lhs, const RawDet& rhs) { return lhs.score > rhs.score; });

    std::vector<bool> suppressed(all_dets.size(), false);
    std::vector<YOLOPv2Detection> kept;

    for (size_t i = 0; i < all_dets.size(); ++i) {
        if (suppressed[i]) continue;
        const RawDet& det = all_dets[i];
        kept.push_back(
            YOLOPv2Detection(det.x1, det.y1, det.x2, det.y2, det.score, YOLOPv2PostProcess::VEHICLE_CLASS_INDEX));

        for (size_t j = i + 1; j < all_dets.size(); ++j) {
            if (suppressed[j]) continue;
            const float inter_x1 = std::max(det.x1, all_dets[j].x1);
            const float inter_y1 = std::max(det.y1, all_dets[j].y1);
            const float inter_x2 = std::min(det.x2, all_dets[j].x2);
            const float inter_y2 = std::min(det.y2, all_dets[j].y2);
            const float iw = std::max(0.0f, inter_x2 - inter_x1);
            const float ih = std::max(0.0f, inter_y2 - inter_y1);
            const float inter_area = iw * ih;
            const float det_area =
                std::max(0.0f, det.x2 - det.x1) * std::max(0.0f, det.y2 - det.y1);
            const float other_area =
                std::max(0.0f, all_dets[j].x2 - all_dets[j].x1) *
                std::max(0.0f, all_dets[j].y2 - all_dets[j].y1);
            const float union_area = det_area + other_area - inter_area;
            if (union_area > 0.0f && (inter_area / union_area) >= nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

YOLOPv2Result YOLOPv2PostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<TensorView> detection_heads;
    TensorView drivable;
    TensorView lane;
    bool has_drivable = false;
    bool has_lane = false;

    for (size_t i = 0; i < outputs.size(); ++i) {
        const TensorView view = make_tensor_view(outputs[i]);
        if (view.channels == 255) {
            detection_heads.push_back(view);
        } else if (view.channels == 2 && !has_drivable) {
            drivable = view;
            has_drivable = true;
        } else if (view.channels == 1 && !has_lane) {
            lane = view;
            has_lane = true;
        }
    }

    if (detection_heads.size() != 3 || !has_drivable || !has_lane) {
        throw std::runtime_error(
            "YOLOPv2PostProcess: expected 3 detection heads, 1 drivable tensor, and 1 lane tensor");
    }

    std::sort(detection_heads.begin(), detection_heads.end(),
              [](const TensorView& lhs, const TensorView& rhs) {
                  return lhs.height * lhs.width > rhs.height * rhs.width;
              });

    std::vector<RawDet> all_dets;
    for (size_t i = 0; i < detection_heads.size(); ++i) {
        std::vector<RawDet> decoded =
            decode_head(detection_heads[i].data, STRIDES[i], detection_heads[i].height,
                        detection_heads[i].width, ANCHORS[i]);
        all_dets.insert(all_dets.end(), decoded.begin(), decoded.end());
    }

    YOLOPv2Result result;
    result.detections = nms_agnostic(all_dets);
    result.mask_height = drivable.height;
    result.mask_width = drivable.width;
    result.drivable_mask.resize(result.mask_height * result.mask_width, 0);
    result.lane_mask.resize(result.mask_height * result.mask_width, 0);

    if (lane.height != result.mask_height || lane.width != result.mask_width) {
        throw std::runtime_error("YOLOPv2PostProcess: segmentation tensor sizes do not match");
    }

    const int mask_plane = result.mask_height * result.mask_width;
    for (int i = 0; i < mask_plane; ++i) {
        const float ch0 = drivable.data[i];
        const float ch1 = drivable.data[mask_plane + i];
        result.drivable_mask[i] = ch1 > ch0 ? 1 : 0;
        result.lane_mask[i] = sigmoid(lane.data[i]) > 0.5f ? 1 : 0;
    }

    return result;
}
