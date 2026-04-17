/**
 * @file ppu_detection_postprocessor.hpp
 * @brief Unified PPU (hardware) anchor-based detection postprocess (YOLOv5-PPU, YOLOv7-PPU)
 *
 * Both models share identical PPU decoding logic; only anchors and thresholds differ.
 */
#ifndef PPU_DETECTION_POSTPROCESSOR_HPP
#define PPU_DETECTION_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>
#include <dxrt/datatype.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common_util.hpp"
#include "postprocess_utils.hpp"

// ============================================================================
// Result type
// ============================================================================
struct PPUDetectionResult {
    std::vector<float> box{};
    float confidence{0.0f};
    int class_id{0};
    std::string class_name{};

    PPUDetectionResult() = default;
    PPUDetectionResult(std::vector<float> box_val, float conf, int cls_id,
                       const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const PPUDetectionResult& other) const {
        return postprocess_utils::compute_iou(box, other.box);
    }

    bool is_invalid(int w, int h) const {
        return box[0] < 0 || box[1] < 0 || box[2] > w || box[3] > h;
    }
};

// Backward-compatible type aliases
using YOLOv5PPUResult = PPUDetectionResult;
using YOLOv7PPUResult = PPUDetectionResult;

// ============================================================================
// Postprocess class
// ============================================================================
class PPUDetectionPostProcess {
public:
    PPUDetectionPostProcess(int input_w, int input_h,
                            float obj_threshold, float score_threshold,
                            float nms_threshold,
                            const std::map<int, std::vector<std::pair<int,int>>>& anchors)
        : input_width_(input_w), input_height_(input_h),
          object_threshold_(obj_threshold), score_threshold_(score_threshold),
          nms_threshold_(nms_threshold), anchors_by_strides_(anchors) {
        ppu_output_names_ = {"BBOX"};
    }

    std::vector<PPUDetectionResult> postprocess(const dxrt::TensorPtrs& outputs) {
        if (outputs.front()->type() != dxrt::DataType::BBOX) {
            std::ostringstream msg;
            msg << "[DXAPP] [ER] PPU Detection PostProcess - Tensor type must be BBOX.\n"
                << "  Unexpected Tensors\n";
            msg << postprocess_utils::format_tensor_shapes_with_type(outputs);
            msg << "Expected dxrt::DataType::BBOX.\n";
            throw std::runtime_error(msg.str());
        }
        auto dets = decoding_ppu_outputs(outputs);
        return apply_nms(dets);
    }

    void set_thresholds(float obj_t, float score_t, float nms_t) {
        if (obj_t >= 0.f && obj_t <= 1.f) object_threshold_ = obj_t;
        if (score_t >= 0.f && score_t <= 1.f) score_threshold_ = score_t;
        if (nms_t >= 0.f && nms_t <= 1.f) nms_threshold_ = nms_t;
    }

    std::string get_config_info() const {
        std::ostringstream oss;
        oss << "PPU Detection PostProcess:\n"
            << "  Input: " << input_width_ << "x" << input_height_ << "\n"
            << "  obj_threshold: " << object_threshold_ << "\n"
            << "  score_threshold: " << score_threshold_ << "\n"
            << "  nms_threshold: " << nms_threshold_ << "\n";
        return oss.str();
    }

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_object_threshold() const { return object_threshold_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    static int get_num_classes() { return num_classes_; }
    const std::map<int, std::vector<std::pair<int,int>>>& get_anchors_by_strides() const { return anchors_by_strides_; }
    const std::vector<std::string>& get_ppu_output_names() const { return ppu_output_names_; }

private:
    int input_width_;
    int input_height_;
    float object_threshold_;
    float score_threshold_;
    float nms_threshold_;
    enum { num_classes_ = 80 };
    std::vector<std::string> ppu_output_names_;
    std::map<int, std::vector<std::pair<int,int>>> anchors_by_strides_;

    std::vector<PPUDetectionResult> decoding_ppu_outputs(const dxrt::TensorPtrs& outputs) const {
        std::vector<PPUDetectionResult> detections;
        auto num_elements = outputs[0]->shape()[1];
        auto* raw = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());
        for (int i = 0; i < num_elements; ++i) {
            auto& bb = raw[i];
            if (bb.score < score_threshold_) continue;

            int stride = std::next(anchors_by_strides_.begin(), bb.layer_idx)->first;
            const auto& anchors = anchors_by_strides_.at(stride);

            float x = (bb.x * 2.0 - 0.5 + bb.grid_x) * stride;
            float y = (bb.y * 2.0 - 0.5 + bb.grid_y) * stride;
            float w = (bb.w * bb.w * 4.0) * anchors[bb.box_idx].first;
            float h = (bb.h * bb.h * 4.0) * anchors[bb.box_idx].second;

            PPUDetectionResult r;
            r.confidence = bb.score;
            r.class_id = bb.label;
            r.class_name = dxapp::common::get_coco_class_name(r.class_id);
            r.box = {x - w/2, y - h/2, x + w/2, y + h/2};
            detections.push_back(std::move(r));
        }
        return detections;
    }

    std::vector<PPUDetectionResult> apply_nms(const std::vector<PPUDetectionResult>& dets) const {
        return postprocess_utils::apply_nms(dets, nms_threshold_);
    }
};

// ============================================================================
// Factory subclasses
// ============================================================================

class YOLOv5PPUPostProcess : public PPUDetectionPostProcess {
public:
    YOLOv5PPUPostProcess(int w = 640, int h = 640,
                         float obj = 0.25f, float score = 0.3f, float nms = 0.45f)
        : PPUDetectionPostProcess(w, h, obj, score, nms,
              {{8,  {{10,13},{16,30},{33,23}}},
               {16, {{30,61},{62,45},{59,119}}},
               {32, {{116,90},{156,198},{373,326}}}}) {}
};

class YOLOv7PPUPostProcess : public PPUDetectionPostProcess {
public:
    YOLOv7PPUPostProcess(int w = 640, int h = 640,
                         float obj = 0.3f, float score = 0.4f, float nms = 0.5f)
        : PPUDetectionPostProcess(w, h, obj, score, nms,
              {{8,  {{12,16},{19,36},{40,28}}},
               {16, {{36,75},{76,55},{72,146}}},
               {32, {{142,110},{192,243},{459,401}}}}) {}
};

/**
 * @brief YOLOv3Tiny PPU postprocess with 2-scale YOLOv3 anchors.
 *   YOLOv3Tiny uses 2 detection scales (stride 16, 32) with 3 anchors each.
 */
class YOLOv3TinyPPUPostProcess : public PPUDetectionPostProcess {
public:
    YOLOv3TinyPPUPostProcess(int w = 416, int h = 416,
                              float obj = 0.25f, float score = 0.25f, float nms = 0.45f)
        : PPUDetectionPostProcess(w, h, obj, score, nms,
              {{16, {{10,14},{23,27},{37,58}}},
               {32, {{81,82},{135,169},{344,319}}}}) {}
};

#endif  // PPU_DETECTION_POSTPROCESSOR_HPP
