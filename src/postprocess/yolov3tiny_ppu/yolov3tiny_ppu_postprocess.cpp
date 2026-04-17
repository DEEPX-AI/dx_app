#include "yolov3tiny_ppu_postprocess.h"

#include <dxrt/datatype.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "common_util.hpp"

// YOLOv3TinyPPUResult methods
float YOLOv3TinyPPUResult::iou(const YOLOv3TinyPPUResult& other) const {
    float x_left   = std::max(box[0], other.box[0]);
    float y_top    = std::max(box[1], other.box[1]);
    float x_right  = std::min(box[2], other.box[2]);
    float y_bottom = std::min(box[3], other.box[3]);

    if (x_right < x_left || y_bottom < y_top) return 0.0f;

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float union_area = area() + other.area() - intersection_area;
    return intersection_area / union_area;
}

bool YOLOv3TinyPPUResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

static void init_anchors(std::map<int, std::vector<std::pair<int, int>>>& anchors_by_strides) {
    // YOLOv3-Tiny anchors:
    //   layer_idx=0 → stride 16: [[10,14],[23,27],[37,58]]
    //   layer_idx=1 → stride 32: [[81,82],[135,169],[344,319]]
    anchors_by_strides = {
        {16, {{10, 14}, {23, 27}, {37, 58}}},
        {32, {{81, 82}, {135, 169}, {344, 319}}}
    };
}

// Constructors
YOLOv3TinyPPUPostProcess::YOLOv3TinyPPUPostProcess(int input_w, int input_h,
                                                   float obj_threshold,
                                                   float score_threshold,
                                                   float nms_threshold)
    : input_width_(input_w), input_height_(input_h),
      object_threshold_(obj_threshold), score_threshold_(score_threshold),
      nms_threshold_(nms_threshold) {
    init_anchors(anchors_by_strides_);
}

YOLOv3TinyPPUPostProcess::YOLOv3TinyPPUPostProcess()
    : input_width_(416), input_height_(416),
      object_threshold_(0.25f), score_threshold_(0.25f), nms_threshold_(0.45f) {
    init_anchors(anchors_by_strides_);
}

// postprocess entry point
std::vector<YOLOv3TinyPPUResult> YOLOv3TinyPPUPostProcess::postprocess(
        const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) return {};
    if (outputs.front()->type() != dxrt::DataType::BBOX) {
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv3Tiny PPU PostProcess - Tensor type must be BBOX.\n"
            << "Expected dxrt::DataType::BBOX.\n";
        throw std::runtime_error(msg.str());
    }
    auto dets = decoding_ppu_outputs(outputs);
    return apply_nms(dets);
}

std::vector<YOLOv3TinyPPUResult> YOLOv3TinyPPUPostProcess::decoding_ppu_outputs(
        const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv3TinyPPUResult> detections;
    auto num_elements = outputs[0]->shape()[1];
    auto* raw_data = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());

    for (int i = 0; i < num_elements; i++) {
        const auto& bb = raw_data[i];
        if (bb.score < score_threshold_) continue;

        // layer_idx 0→stride 16, 1→stride 32
        // anchors_by_strides_ is ordered: first entry index 0, second entry index 1
        int layer = static_cast<int>(bb.layer_idx);
        int num_layers = static_cast<int>(anchors_by_strides_.size());
        if (layer < 0 || layer >= num_layers) layer = 0;

        auto it = std::next(anchors_by_strides_.begin(), layer);
        int stride = it->first;
        const auto& anchors = it->second;

        int box_idx = static_cast<int>(bb.box_idx);
        if (box_idx < 0 || box_idx >= static_cast<int>(anchors.size())) box_idx = 0;

        float cx = (bb.x * 2.f - 0.5f + static_cast<float>(bb.grid_x)) * stride;
        float cy = (bb.y * 2.f - 0.5f + static_cast<float>(bb.grid_y)) * stride;
        float w  = (bb.w * bb.w * 4.f) * static_cast<float>(anchors[box_idx].first);
        float h  = (bb.h * bb.h * 4.f) * static_cast<float>(anchors[box_idx].second);

        YOLOv3TinyPPUResult result;
        result.confidence = bb.score;
        result.class_id   = static_cast<int>(bb.label);
        result.class_name = dxapp::common::get_coco_class_name(result.class_id);
        result.box = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f};
        detections.push_back(std::move(result));
    }
    return detections;
}

std::vector<YOLOv3TinyPPUResult> YOLOv3TinyPPUPostProcess::apply_nms(
        const std::vector<YOLOv3TinyPPUResult>& detections) const {
    if (detections.empty()) return {};

    std::vector<YOLOv3TinyPPUResult> sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
              [](const YOLOv3TinyPPUResult& a, const YOLOv3TinyPPUResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<YOLOv3TinyPPUResult> result;

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

void YOLOv3TinyPPUPostProcess::set_thresholds(float obj_threshold, float score_threshold,
                                               float nms_threshold) {
    if (obj_threshold   >= 0.f && obj_threshold   <= 1.f) object_threshold_  = obj_threshold;
    if (score_threshold >= 0.f && score_threshold <= 1.f) score_threshold_   = score_threshold;
    if (nms_threshold   >= 0.f && nms_threshold   <= 1.f) nms_threshold_     = nms_threshold;
}

std::string YOLOv3TinyPPUPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv3Tiny PPU PostProcess Configuration:\n"
        << "  Input: " << input_width_ << "x" << input_height_ << "\n"
        << "  obj_threshold: "   << object_threshold_  << "\n"
        << "  score_threshold: " << score_threshold_   << "\n"
        << "  nms_threshold: "   << nms_threshold_     << "\n"
        << "  Anchors: layer_idx=0→stride 16, layer_idx=1→stride 32\n";
    return oss.str();
}
