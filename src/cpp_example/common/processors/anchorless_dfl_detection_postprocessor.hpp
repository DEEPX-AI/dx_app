/**
 * @file anchorless_dfl_detection_postprocessor.hpp
 * @brief Unified anchor-free DFL-based YOLO detection postprocess
 *        (YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLOv26)
 *
 * v8/v9/v11/v12 share 100% identical logic (transposed-xywh CPU decode).
 * v10/v26 differ only in CPU decode (end-to-end: [1,300,6] layout).
 * NPU decode is the same for all.
 */
#ifndef ANCHORLESS_DFL_DETECTION_POSTPROCESSOR_HPP
#define ANCHORLESS_DFL_DETECTION_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common_util.hpp"
#include "postprocess_utils.hpp"

// ============================================================================
// CPU decode mode
// ============================================================================
enum class AnchorlessCpuDecodeMode {
    TRANSPOSED_XYWH,  // [1, 84, 8400] — v8 / v9 / v11 / v12
    END_TO_END,        // [1, 300, 6]   — v10 / v26
};

// ============================================================================
// Result type
// ============================================================================
struct AnchorlessYOLOResult {
    std::vector<float> box{};
    float confidence{0.0f};
    int class_id{0};
    std::string class_name{};

    AnchorlessYOLOResult() = default;
    AnchorlessYOLOResult(std::vector<float> box_val, float conf, int cls_id,
                         const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const AnchorlessYOLOResult& other) const {
        return postprocess_utils::compute_iou(box, other.box);
    }

    bool is_invalid(int w, int h) const {
        return box[0] < 0 || box[1] < 0 || box[2] > w || box[3] > h;
    }
};

// Backward-compatible type aliases
using YOLOv8Result  = AnchorlessYOLOResult;
using YOLOv9Result  = AnchorlessYOLOResult;
using YOLOv10Result = AnchorlessYOLOResult;
using YOLOv11Result = AnchorlessYOLOResult;
using YOLOv12Result = AnchorlessYOLOResult;
using YOLOv26Result = AnchorlessYOLOResult;

// ============================================================================
// Postprocess class
// ============================================================================
class AnchorlessYOLOPostProcess {
public:
    AnchorlessYOLOPostProcess(int input_w, int input_h,
                              float score_threshold, float nms_threshold,
                              bool is_ort_configured,
                              AnchorlessCpuDecodeMode cpu_mode = AnchorlessCpuDecodeMode::TRANSPOSED_XYWH)
        : input_width_(input_w), input_height_(input_h),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          is_ort_configured_(is_ort_configured), cpu_mode_(cpu_mode) {
        cpu_output_names_ = {"output0"};
        npu_output_names_ = {"/model.22/dfl/conv/Conv_output_0",
                             "/model.22/Sigmoid_output_0"};
        anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
    }

    std::vector<AnchorlessYOLOResult> postprocess(const dxrt::TensorPtrs& outputs) {
        auto aligned = align_tensors(outputs);
        if (aligned.empty()) {
            std::ostringstream msg;
            msg << "[DXAPP] [ER] AnchorlessYOLOPostProcess - Aligned outputs are empty.\n"
                << "  Unexpected shape\n";
            msg << postprocess_utils::format_tensor_shapes(outputs);
            msg << "Please re-compile the model with the correct output configuration.\n";
            throw std::runtime_error(msg.str());
        }

        std::vector<AnchorlessYOLOResult> dets;
        if (is_ort_configured_) {
            dets = decoding_cpu_outputs(aligned);
        } else {
            dets = decoding_npu_outputs(aligned);
        }
        return apply_nms(dets);
    }

    dxrt::TensorPtrs align_tensors(const dxrt::TensorPtrs& outputs) const {
        dxrt::TensorPtrs aligned;
        if (is_ort_configured_) {
            for (const auto& o : outputs) {
                if (o->shape().size() == 3) { aligned.push_back(o); break; }
            }
            return aligned;
        }
        if (outputs.size() == 6) {
            std::vector<dxrt::TensorPtr> cls_out, reg_out;
            for (const auto& o : outputs) {
                if (o->shape().size() != 4) continue;
                if (o->shape()[1] == num_classes_) cls_out.push_back(o);
                else if (o->shape()[1] == 64) reg_out.push_back(o);
            }
            if (cls_out.size() == 3 && reg_out.size() == 3) {
                auto cmp = [](const dxrt::TensorPtr& a, const dxrt::TensorPtr& b) {
                    return a->shape()[2] > b->shape()[2];
                };
                std::sort(cls_out.begin(), cls_out.end(), cmp);
                std::sort(reg_out.begin(), reg_out.end(), cmp);
                for (size_t i = 0; i < 3; ++i) {
                    aligned.push_back(reg_out[i]);
                    aligned.push_back(cls_out[i]);
                }
                return aligned;
            }
        }
        return aligned;
    }

    void set_thresholds(float score_t, float nms_t) {
        if (score_t >= 0.f && score_t <= 1.f) score_threshold_ = score_t;
        if (nms_t >= 0.f && nms_t <= 1.f) nms_threshold_ = nms_t;
    }

    std::string get_config_info() const {
        std::ostringstream oss;
        oss << "AnchorlessYOLO PostProcess Configuration:\n"
            << "  Input: " << input_width_ << "x" << input_height_ << "\n"
            << "  score_threshold: " << score_threshold_ << "\n"
            << "  nms_threshold: " << nms_threshold_ << "\n"
            << "  ORT: " << (is_ort_configured_ ? "Yes" : "No") << "\n"
            << "  CPU mode: " << (cpu_mode_ == AnchorlessCpuDecodeMode::TRANSPOSED_XYWH
                                  ? "TRANSPOSED_XYWH" : "END_TO_END") << "\n";
        return oss.str();
    }

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    bool get_is_ort_configured() const { return is_ort_configured_; }
    static int get_num_classes() { return num_classes_; }
    const std::map<int, std::vector<std::pair<int,int>>>& get_anchors_by_strides() const { return anchors_by_strides_; }
    const std::vector<std::string>& get_cpu_output_names() const { return cpu_output_names_; }
    const std::vector<std::string>& get_npu_output_names() const { return npu_output_names_; }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    enum { num_classes_ = 80 };
    bool is_ort_configured_;
    AnchorlessCpuDecodeMode cpu_mode_;
    std::vector<std::string> cpu_output_names_;
    std::vector<std::string> npu_output_names_;
    std::map<int, std::vector<std::pair<int,int>>> anchors_by_strides_;

    static float sigmoid(float x) { return postprocess_utils::sigmoid(x); }

    // Helper: softmax-weighted DFL distance for regression direction k at grid position sp.
    float compute_dfl_dist(const float* reg_data, int k, int num_grid, int sp) const {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int d = 0; d < 16; ++d) {
            float v = reg_data[(k * 16 + d) * num_grid + sp];
            if (v > max_val) max_val = v;
        }
        float exp_sum = 0.f, weighted_sum = 0.f;
        for (int d = 0; d < 16; ++d) {
            float e = std::exp(reg_data[(k * 16 + d) * num_grid + sp] - max_val);
            exp_sum     += e;
            weighted_sum += e * d;
        }
        return weighted_sum / exp_sum;
    }

    // Helper: find the best class at grid position sp.
    // Populates max_cls / max_conf; returns true when a class exceeds the threshold.
    bool find_best_class(const float* cls_data, int num_grid, int sp,
                         int& max_cls, float& max_conf) const {
        max_cls  = -1;
        max_conf = score_threshold_;
        for (int c = 0; c < num_classes_; ++c) {
            float conf = cls_data[c * num_grid + sp];
            if (conf > max_conf) { max_conf = conf; max_cls = c; }
        }
        return max_cls != -1;
    }

    // ---- NPU decode: per-stride grid decoding ----
    void decodeStrideGrid(const float* reg_data, const float* cls_data,
                          int H, int W, int stride,
                          std::vector<AnchorlessYOLOResult>& detections) const {
        int num_grid = H * W;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int sp = h * W + w;
                int   max_cls  = -1;
                float max_conf = 0.f;
                if (!find_best_class(cls_data, num_grid, sp, max_cls, max_conf)) continue;

                float dist[4];
                for (int k = 0; k < 4; ++k)
                    dist[k] = compute_dfl_dist(reg_data, k, num_grid, sp);

                float ax = w + 0.5f, ay = h + 0.5f;
                AnchorlessYOLOResult r;
                r.confidence = max_conf;
                r.class_id   = max_cls;
                r.class_name = dxapp::common::get_coco_class_name(max_cls);
                r.box = {(ax - dist[0]) * stride, (ay - dist[1]) * stride,
                         (ax + dist[2]) * stride, (ay + dist[3]) * stride};
                detections.push_back(std::move(r));
            }
        }
    }

    // ---- NPU decode (shared by all 6 models) ----
    std::vector<AnchorlessYOLOResult> decoding_npu_outputs(const dxrt::TensorPtrs& outputs) const {
        std::vector<AnchorlessYOLOResult> detections;

        if (outputs.size() == 6) {
            for (size_t i = 0; i < 3; ++i) {
                const auto& reg_tensor = outputs[2 * i];
                const auto& cls_tensor = outputs[2 * i + 1];
                auto reg_data = static_cast<const float*>(reg_tensor->data());
                auto cls_data = static_cast<const float*>(cls_tensor->data());

                auto H = static_cast<int>(cls_tensor->shape()[2]);
                auto W = static_cast<int>(cls_tensor->shape()[3]);
                int stride = input_width_ / W;
                decodeStrideGrid(reg_data, cls_data, H, W, stride, detections);
            }
        }
        return detections;
    }

    // ---- CPU decode (differs per family) ----
    std::vector<AnchorlessYOLOResult> decoding_cpu_outputs(const dxrt::TensorPtrs& outputs) const {
        if (cpu_mode_ == AnchorlessCpuDecodeMode::END_TO_END)
            return decoding_cpu_e2e(outputs);
        return decoding_cpu_transposed(outputs);
    }

    // v8/v9/v11/v12: [1, 84, 8400] transposed layout
    std::vector<AnchorlessYOLOResult> decoding_cpu_transposed(const dxrt::TensorPtrs& outputs) const {
        std::vector<AnchorlessYOLOResult> detections;

        auto find_best_transposed_class = [&](const float* data, int num_dets, int i)
            -> std::pair<int, float> {
            int best_cls = -1;
            float best_conf = score_threshold_;
            for (int c = 0; c < num_classes_; ++c) {
                float conf = data[(4 + c) * num_dets + i];
                if (conf > best_conf) { best_conf = conf; best_cls = c; }
            }
            return {best_cls, best_conf};
        };

        for (const auto& output : outputs) {
            auto data = static_cast<const float*>(output->data());
            auto num_dets = static_cast<int>(output->shape()[2]);
            for (int i = 0; i < num_dets; ++i) {
                auto [max_cls, max_conf] = find_best_transposed_class(data, num_dets, i);
                if (max_cls == -1) continue;

                float bx[4];
                for (int j = 0; j < 4; ++j) bx[j] = data[j * num_dets + i];

                AnchorlessYOLOResult r;
                r.confidence = max_conf;
                r.class_id = max_cls;
                r.class_name = dxapp::common::get_coco_class_name(max_cls);
                r.box = {bx[0] - bx[2]/2, bx[1] - bx[3]/2,
                         bx[0] + bx[2]/2, bx[1] + bx[3]/2};
                detections.push_back(std::move(r));
            }
        }
        return detections;
    }

    // v10/v26: [1, 300, 6] end-to-end layout
    std::vector<AnchorlessYOLOResult> decoding_cpu_e2e(const dxrt::TensorPtrs& outputs) const {
        std::vector<AnchorlessYOLOResult> detections;
        if (outputs.empty()) return detections;
        const auto& tensor = outputs[0];
        const float* data = static_cast<const float*>(tensor->data());
        const auto& shape = tensor->shape();
        if (shape.size() != 3) return detections;
        int N = static_cast<int>(shape[1]);
        int stride = static_cast<int>(shape[2]);
        for (int i = 0; i < N; ++i) {
            const float* det = data + i * stride;
            if (det[4] < score_threshold_) continue;
            int cls = static_cast<int>(det[5]);
            if (cls < 0 || cls >= num_classes_) continue;
            AnchorlessYOLOResult r;
            r.confidence = det[4];
            r.class_id = cls;
            r.class_name = dxapp::common::get_coco_class_name(cls);
            r.box = {det[0], det[1], det[2], det[3]};
            detections.push_back(std::move(r));
        }
        return detections;
    }

    std::vector<AnchorlessYOLOResult> apply_nms(const std::vector<AnchorlessYOLOResult>& dets) const {
        return postprocess_utils::apply_nms(dets, nms_threshold_);
    }
};

// ============================================================================
// Factory subclasses — replicate old constructors
// ============================================================================

class YOLOv8PostProcess : public AnchorlessYOLOPostProcess {
public:
    YOLOv8PostProcess(int w = 640, int h = 640,
                      float score = 0.45f, float nms = 0.4f, bool ort = false)
        : AnchorlessYOLOPostProcess(w, h, score, nms, ort,
              AnchorlessCpuDecodeMode::TRANSPOSED_XYWH) {}
};

class YOLOv9PostProcess : public AnchorlessYOLOPostProcess {
public:
    YOLOv9PostProcess(int w = 640, int h = 640,
                      float score = 0.45f, float nms = 0.4f, bool ort = false)
        : AnchorlessYOLOPostProcess(w, h, score, nms, ort,
              AnchorlessCpuDecodeMode::TRANSPOSED_XYWH) {}
};

class YOLOv11PostProcess : public AnchorlessYOLOPostProcess {
public:
    YOLOv11PostProcess(int w = 640, int h = 640,
                       float score = 0.45f, float nms = 0.4f, bool ort = false)
        : AnchorlessYOLOPostProcess(w, h, score, nms, ort,
              AnchorlessCpuDecodeMode::TRANSPOSED_XYWH) {}
};

class YOLOv12PostProcess : public AnchorlessYOLOPostProcess {
public:
    YOLOv12PostProcess(int w = 640, int h = 640,
                       float score = 0.45f, float nms = 0.4f, bool ort = false)
        : AnchorlessYOLOPostProcess(w, h, score, nms, ort,
              AnchorlessCpuDecodeMode::TRANSPOSED_XYWH) {}
};

class YOLOv10PostProcess : public AnchorlessYOLOPostProcess {
public:
    YOLOv10PostProcess(int w = 640, int h = 640,
                       float score = 0.45f, float nms = 0.4f, bool ort = false)
        : AnchorlessYOLOPostProcess(w, h, score, nms, ort,
              AnchorlessCpuDecodeMode::END_TO_END) {}
};

class YOLOv26PostProcess : public AnchorlessYOLOPostProcess {
public:
    YOLOv26PostProcess(int w = 640, int h = 640,
                       float score = 0.45f, float nms = 0.4f, bool ort = false)
        : AnchorlessYOLOPostProcess(w, h, score, nms, ort,
              AnchorlessCpuDecodeMode::END_TO_END) {}
};

#endif  // ANCHORLESS_DFL_DETECTION_POSTPROCESSOR_HPP
