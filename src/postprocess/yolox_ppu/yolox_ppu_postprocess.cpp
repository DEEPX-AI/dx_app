#include "yolox_ppu_postprocess.h"

#include <dxrt/datatype.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "common_util.hpp"

// Definition of static constexpr members
constexpr int YOLOXPPUPostProcess::STRIDES[YOLOXPPUPostProcess::NUM_STRIDES];

// YOLOXPPUResult methods
float YOLOXPPUResult::iou(const YOLOXPPUResult& other) const {
    float x_left   = std::max(box[0], other.box[0]);
    float y_top    = std::max(box[1], other.box[1]);
    float x_right  = std::min(box[2], other.box[2]);
    float y_bottom = std::min(box[3], other.box[3]);

    if (x_right < x_left || y_bottom < y_top) return 0.0f;

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float union_area = area() + other.area() - intersection_area;
    return intersection_area / union_area;
}

bool YOLOXPPUResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructors
YOLOXPPUPostProcess::YOLOXPPUPostProcess(int input_w, int input_h,
                                         float obj_threshold, float score_threshold,
                                         float nms_threshold)
    : input_width_(input_w), input_height_(input_h),
      object_threshold_(obj_threshold), score_threshold_(score_threshold),
      nms_threshold_(nms_threshold) {}

YOLOXPPUPostProcess::YOLOXPPUPostProcess()
    : input_width_(640), input_height_(640),
      object_threshold_(0.25f), score_threshold_(0.25f), nms_threshold_(0.45f) {}

// postprocess entry point
std::vector<YOLOXPPUResult> YOLOXPPUPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) return {};
    if (outputs.front()->type() != dxrt::DataType::BBOX) {
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOX PPU PostProcess - Tensor type must be BBOX.\n"
            << "Expected dxrt::DataType::BBOX.\n";
        throw std::runtime_error(msg.str());
    }
    auto dets = decoding_ppu_outputs(outputs);
    return apply_nms(dets);
}

std::vector<YOLOXPPUResult> YOLOXPPUPostProcess::decoding_ppu_outputs(
        const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOXPPUResult> detections;
    auto num_elements = outputs[0]->shape()[1];
    auto* raw_data = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());

    for (int i = 0; i < num_elements; i++) {
        const auto& bb = raw_data[i];
        if (bb.score < score_threshold_) continue;

        // layer_idx selects stride: 0→8, 1→16, 2→32
        int layer = static_cast<int>(bb.layer_idx);
        if (layer < 0 || layer >= NUM_STRIDES) layer = 0;
        float stride = static_cast<float>(STRIDES[layer]);

        // Anchor-free YOLOX decode
        float cx = (bb.x + static_cast<float>(bb.grid_x)) * stride;
        float cy = (bb.y + static_cast<float>(bb.grid_y)) * stride;
        float tw = std::max(-10.f, std::min(10.f, bb.w));  // clamp to prevent exp overflow
        float th = std::max(-10.f, std::min(10.f, bb.h));
        float w  = std::exp(tw) * stride;
        float h  = std::exp(th) * stride;

        YOLOXPPUResult result;
        result.confidence = bb.score;
        result.class_id   = static_cast<int>(bb.label);
        result.class_name = dxapp::common::get_coco_class_name(result.class_id);
        result.box = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f};
        detections.push_back(std::move(result));
    }
    return detections;
}

std::vector<YOLOXPPUResult> YOLOXPPUPostProcess::apply_nms(
        const std::vector<YOLOXPPUResult>& detections) const {
    if (detections.empty()) return {};

    std::vector<YOLOXPPUResult> sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
              [](const YOLOXPPUResult& a, const YOLOXPPUResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<YOLOXPPUResult> result;

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (!suppressed[j] && sorted[i].iou(sorted[j]) > nms_threshold_)
                suppressed[j] = true;
        }
    }
    return result;
}

void YOLOXPPUPostProcess::set_thresholds(float obj_threshold, float score_threshold,
                                          float nms_threshold) {
    if (obj_threshold   >= 0.f && obj_threshold   <= 1.f) object_threshold_  = obj_threshold;
    if (score_threshold >= 0.f && score_threshold <= 1.f) score_threshold_   = score_threshold;
    if (nms_threshold   >= 0.f && nms_threshold   <= 1.f) nms_threshold_     = nms_threshold;
}

std::string YOLOXPPUPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOX PPU PostProcess Configuration:\n"
        << "  Input: " << input_width_ << "x" << input_height_ << "\n"
        << "  obj_threshold: " << object_threshold_ << "\n"
        << "  score_threshold: " << score_threshold_ << "\n"
        << "  nms_threshold: " << nms_threshold_ << "\n"
        << "  Strides: 8, 16, 32 (indexed by layer_idx)\n";
    return oss.str();
}
