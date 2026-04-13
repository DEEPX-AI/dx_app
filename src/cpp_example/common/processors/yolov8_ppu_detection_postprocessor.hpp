/**
 * @file yolov8_ppu_detection_postprocessor.hpp
 * @brief YOLOv8 PPU (hardware) anchor-free detection postprocess
 *
 * Unlike YOLOv5/v7 PPU which use anchor-based decoding, YOLOv8 PPU uses
 * anchor-free decoding where BBOX outputs contain direct x, y, w, h values.
 */
#ifndef YOLOV8_PPU_DETECTION_POSTPROCESSOR_HPP
#define YOLOV8_PPU_DETECTION_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>
#include <dxrt/datatype.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "common_util.hpp"
#include "postprocess_utils.hpp"

// ============================================================================
// Result type
// ============================================================================
struct YOLOv8PPUResult {
    std::vector<float> box{};
    float confidence{0.0f};
    int class_id{0};
    std::string class_name{};

    YOLOv8PPUResult() = default;
    YOLOv8PPUResult(std::vector<float> box_val, float conf, int cls_id,
                     const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const YOLOv8PPUResult& other) const {
        return postprocess_utils::compute_iou(box, other.box);
    }

    bool is_invalid(int w, int h) const {
        return box[0] < 0 || box[1] < 0 || box[2] > w || box[3] > h;
    }
};

// ============================================================================
// Postprocess class — anchor-free PPU decoding
// ============================================================================
class YOLOv8PPUPostProcess {
public:
    YOLOv8PPUPostProcess(int input_w = 640, int input_h = 640,
                          float score_threshold = 0.4f,
                          float nms_threshold = 0.5f)
        : input_width_(input_w), input_height_(input_h),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold) {
        ppu_output_names_ = {"BBOX"};
    }

    std::vector<YOLOv8PPUResult> postprocess(const dxrt::TensorPtrs& outputs) {
        if (outputs.front()->type() != dxrt::DataType::BBOX) {
            std::ostringstream msg;
            msg << "[DXAPP] [ER] YOLOv8 PPU PostProcess - Tensor type must be BBOX.\n"
                << "  Unexpected Tensors\n";
            msg << postprocess_utils::format_tensor_shapes_with_type(outputs);
            msg << "Expected dxrt::DataType::BBOX.\n";
            throw std::runtime_error(msg.str());
        }
        auto dets = decoding_ppu_outputs(outputs);
        return apply_nms(dets);
    }

    void set_thresholds(float score_t, float nms_t) {
        if (score_t >= 0.f && score_t <= 1.f) score_threshold_ = score_t;
        if (nms_t >= 0.f && nms_t <= 1.f) nms_threshold_ = nms_t;
    }

    std::string get_config_info() const {
        std::ostringstream oss;
        oss << "YOLOv8 PPU PostProcess:\n"
            << "  Input: " << input_width_ << "x" << input_height_ << "\n"
            << "  score_threshold: " << score_threshold_ << "\n"
            << "  nms_threshold: " << nms_threshold_ << "\n";
        return oss.str();
    }

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    static int get_num_classes() { return num_classes_; }
    const std::vector<std::string>& get_ppu_output_names() const { return ppu_output_names_; }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    enum { num_classes_ = 80 };
    std::vector<std::string> ppu_output_names_;

    std::vector<YOLOv8PPUResult> decoding_ppu_outputs(const dxrt::TensorPtrs& outputs) const {
        std::vector<YOLOv8PPUResult> detections;

        if (outputs.empty() || outputs[0]->shape().size() < 2) {
            throw std::runtime_error("[DXAPP] [ER] YOLOv8 PPU decoding - Invalid output shape");
        }

        auto num_elements = outputs[0]->shape()[1];
        auto* raw = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());
        for (int i = 0; i < num_elements; ++i) {
            auto& bb = raw[i];
            if (bb.score < score_threshold_) continue;

            // YOLOv8 PPU: anchor-free, direct x/y/w/h
            float x = bb.x;
            float y = bb.y;
            float w = bb.w;
            float h = bb.h;

            YOLOv8PPUResult r;
            r.confidence = bb.score;
            r.class_id = bb.label;
            r.class_name = dxapp::common::get_coco_class_name(r.class_id);
            r.box = {x - w / 2, y - h / 2, x + w / 2, y + h / 2};
            detections.push_back(std::move(r));
        }
        return detections;
    }

    std::vector<YOLOv8PPUResult> apply_nms(const std::vector<YOLOv8PPUResult>& dets) const {
        return postprocess_utils::apply_nms(dets, nms_threshold_);
    }
};

// ============================================================================
// YOLOX PPU — anchor-free detection with grid-based decoding
//   cx = (tx + grid_x) * stride
//   cy = (ty + grid_y) * stride
//   w  = exp(tw) * stride
//   h  = exp(th) * stride
// ============================================================================
class YOLOXPPUPostProcess {
public:
    YOLOXPPUPostProcess(int input_w = 640, int input_h = 640,
                         float score_threshold = 0.25f,
                         float nms_threshold = 0.45f)
        : input_width_(input_w), input_height_(input_h),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold) {}

    std::vector<YOLOv8PPUResult> postprocess(const dxrt::TensorPtrs& outputs) {
        if (outputs.front()->type() != dxrt::DataType::BBOX) {
            std::ostringstream msg;
            msg << "[DXAPP] [ER] YOLOX PPU PostProcess - Tensor type must be BBOX.\n";
            msg << postprocess_utils::format_tensor_shapes_with_type(outputs);
            throw std::runtime_error(msg.str());
        }
        auto dets = decoding_ppu_outputs(outputs);
        return apply_nms(dets);
    }

    void set_thresholds(float score_t, float nms_t) {
        if (score_t >= 0.f && score_t <= 1.f) score_threshold_ = score_t;
        if (nms_t >= 0.f && nms_t <= 1.f) nms_threshold_ = nms_t;
    }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    static constexpr int STRIDES[3] = {8, 16, 32};

    std::vector<YOLOv8PPUResult> decoding_ppu_outputs(const dxrt::TensorPtrs& outputs) const {
        std::vector<YOLOv8PPUResult> detections;
        if (outputs.empty() || outputs[0]->shape().size() < 2) return detections;

        auto num_elements = outputs[0]->shape()[1];
        auto* raw = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());

        for (int i = 0; i < num_elements; ++i) {
            auto& bb = raw[i];
            if (bb.score < score_threshold_) continue;

            int layer = bb.layer_idx;
            int stride = (layer < 3) ? STRIDES[layer] : STRIDES[0];

            // YOLOX anchor-free: tx/ty are raw grid offsets, tw/th need exp
            float cx = (bb.x + static_cast<float>(bb.grid_x)) * stride;
            float cy = (bb.y + static_cast<float>(bb.grid_y)) * stride;
            float tw = std::min(bb.w, 10.0f);  // clamp to avoid exp overflow
            float th = std::min(bb.h, 10.0f);
            float w = std::exp(tw) * stride;
            float h = std::exp(th) * stride;

            YOLOv8PPUResult r;
            r.confidence = bb.score;
            r.class_id = bb.label;
            r.class_name = dxapp::common::get_coco_class_name(r.class_id);
            r.box = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
            detections.push_back(std::move(r));
        }
        return detections;
    }

    std::vector<YOLOv8PPUResult> apply_nms(const std::vector<YOLOv8PPUResult>& dets) const {
        return postprocess_utils::apply_nms(dets, nms_threshold_);
    }
};

constexpr int YOLOXPPUPostProcess::STRIDES[3];

#endif  // YOLOV8_PPU_DETECTION_POSTPROCESSOR_HPP
