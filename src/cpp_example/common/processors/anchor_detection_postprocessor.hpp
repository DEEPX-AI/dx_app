/**
 * @file anchor_detection_postprocessor.hpp
 * @brief Unified anchor-based YOLO detection postprocess (YOLOv5, YOLOv7, YOLOX)
 *
 * These models share identical logic; only anchors, thresholds, and tensor names differ.
 * The class is parameterised through the constructor so a single implementation
 * serves all three model families.
 */
#ifndef ANCHOR_DETECTION_POSTPROCESSOR_HPP
#define ANCHOR_DETECTION_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>

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
struct AnchorYOLOResult {
    std::vector<float> box{};
    float confidence{0.0f};
    int class_id{0};
    std::string class_name{};

    AnchorYOLOResult() = default;
    AnchorYOLOResult(std::vector<float> box_val, float conf, int cls_id,
                     const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const AnchorYOLOResult& other) const {
        return postprocess_utils::compute_iou(box, other.box);
    }

    bool is_invalid(int image_width, int image_height) const {
        return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
    }
};

// Backward-compatible type aliases
using YOLOv5Result  = AnchorYOLOResult;
using YOLOv7Result  = AnchorYOLOResult;
using YOLOXResult   = AnchorYOLOResult;

// ============================================================================
// Postprocess class
// ============================================================================
class AnchorYOLOPostProcess {
public:
    AnchorYOLOPostProcess(int input_w, int input_h,
                          float obj_threshold, float score_threshold,
                          float nms_threshold, bool is_ort_configured,
                          const std::vector<std::string>& cpu_output_names,
                          const std::vector<std::string>& npu_output_names,
                          const std::map<int, std::vector<std::pair<int,int>>>& anchors,
                          bool npu_supported = true)
        : input_width_(input_w), input_height_(input_h),
          object_threshold_(obj_threshold), score_threshold_(score_threshold),
          nms_threshold_(nms_threshold), is_ort_configured_(is_ort_configured),
          cpu_output_names_(cpu_output_names), npu_output_names_(npu_output_names),
          anchors_by_strides_(anchors), npu_supported_(npu_supported) {
        if (!is_ort_configured_ && !npu_supported_) {
            throw std::invalid_argument(
                "ORT-OFF output postprocessing is not supported for this model.\n"
                "Please build dxrt with USE_ORT=ON");
        }
    }

    std::vector<AnchorYOLOResult> postprocess(const dxrt::TensorPtrs& outputs) {
        auto aligned = align_tensors(outputs);
        std::vector<AnchorYOLOResult> dets;
        // Determine decode path: if aligned has a 3D tensor, use CPU (decoded) path;
        // otherwise use NPU (raw) path for NCHW/NHWC/5D tensors
        bool use_cpu_path = is_ort_configured_;
        if (use_cpu_path && !aligned.empty()) {
            bool has_3d = false;
            for (const auto& t : aligned) {
                if (t->shape().size() == 3) { has_3d = true; break; }
            }
            if (!has_3d) use_cpu_path = false;  // Fall back to NPU decode for raw outputs
        }
        if (use_cpu_path) {
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
            // If no 3D tensor found, check for 5D raw or NHWC 4D → fall through to NPU path
            if (!aligned.empty()) return aligned;
        }
        for (const auto& as : anchors_by_strides_) {
            for (const auto& o : outputs) {
                auto shape = o->shape();
                int stride = as.first;
                int expected_w = input_width_ / stride;
                int expected_h = input_height_ / stride;
                int expected_c = static_cast<int>((num_classes_ + 5) * as.second.size());
                // NCHW: [1, C, H, W]
                if (shape.size() == 4 &&
                    shape[1] == expected_c &&
                    shape[2] == expected_h &&
                    shape[3] == expected_w) {
                    aligned.push_back(o);
                    break;
                }
                // NHWC: [1, H, W, C]
                if (shape.size() == 4 &&
                    shape[1] == expected_h &&
                    shape[2] == expected_w &&
                    shape[3] == expected_c) {
                    aligned.push_back(o);
                    break;
                }
                // 5D raw: [1, num_anchors, H, W, (5+num_classes)]
                if (shape.size() == 5 &&
                    shape[1] == static_cast<int64_t>(as.second.size()) &&
                    shape[2] == expected_h &&
                    shape[3] == expected_w &&
                    shape[4] == (num_classes_ + 5)) {
                    aligned.push_back(o);
                    break;
                }
            }
        }
        if (aligned.empty()) {
            std::cerr << "[DXAPP] [ERROR] Failed to align output tensors." << std::endl;
            return outputs;
        }
        return aligned;
    }

    void set_thresholds(float obj_t, float score_t, float nms_t) {
        if (obj_t >= 0.f && obj_t <= 1.f) object_threshold_ = obj_t;
        if (score_t >= 0.f && score_t <= 1.f) score_threshold_ = score_t;
        if (nms_t >= 0.f && nms_t <= 1.f) nms_threshold_ = nms_t;
    }

    std::string get_config_info() const {
        std::ostringstream oss;
        oss << "AnchorYOLO PostProcess Configuration:\n"
            << "  Input: " << input_width_ << "x" << input_height_ << "\n"
            << "  obj_threshold: " << object_threshold_ << "\n"
            << "  score_threshold: " << score_threshold_ << "\n"
            << "  nms_threshold: " << nms_threshold_ << "\n"
            << "  ORT: " << (is_ort_configured_ ? "Yes" : "No") << "\n";
        return oss.str();
    }

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_object_threshold() const { return object_threshold_; }
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
    float object_threshold_;
    float score_threshold_;
    float nms_threshold_;
    enum { num_classes_ = 80 };
    bool is_ort_configured_;
    bool npu_supported_;
    std::vector<std::string> cpu_output_names_;
    std::vector<std::string> npu_output_names_;
    std::map<int, std::vector<std::pair<int,int>>> anchors_by_strides_;

    static float sigmoid(float x) { return postprocess_utils::sigmoid(x); }

    // Helper: find best class from detection row scores
    std::pair<int, float> find_best_class_score_(const float* det, float obj) const {
        int best_cls = -1;
        float best_conf = score_threshold_;
        for (int c = 0; c < num_classes_; ++c) {
            float conf = obj * det[5 + c];
            if (conf > best_conf) { best_conf = conf; best_cls = c; }
        }
        return {best_cls, best_conf};
    }

    // Helper: decode one grid cell from a single NPU anchor output.
    // Populates `out` and returns true when a valid detection is found.
    bool decode_npu_anchor_cell(const float* data, int a, int gx, int gy,
                                int gx_sz, int gy_sz, int aw, int ah, int stride,
                                AnchorYOLOResult& out) const {
        int obj_idx = ((a * (num_classes_ + 5)) + 4) * gx_sz * gy_sz + gy * gx_sz + gx;
        float obj = sigmoid(data[obj_idx]);
        if (obj < object_threshold_) return false;

        int max_cls = -1;
        float max_conf = score_threshold_;
        for (int c = 0; c < num_classes_; ++c) {
            int ci = ((a * (num_classes_ + 5)) + 5 + c) * gx_sz * gy_sz + gy * gx_sz + gx;
            float conf = obj * sigmoid(data[ci]);
            if (conf > max_conf) { max_conf = conf; max_cls = c; }
        }
        if (max_cls == -1) return false;

        float bx[4];
        for (int i = 0; i < 4; ++i) {
            int bi = ((a * (num_classes_ + 5)) + i) * gx_sz * gy_sz + gy * gx_sz + gx;
            bx[i] = data[bi];
        }
        float cx = (sigmoid(bx[0]) * 2.f - 0.5f + gx) * stride;
        float cy = (sigmoid(bx[1]) * 2.f - 0.5f + gy) * stride;
        float w  = std::pow(sigmoid(bx[2]) * 2.f, 2.f) * aw;
        float h  = std::pow(sigmoid(bx[3]) * 2.f, 2.f) * ah;

        out.confidence = max_conf;
        out.class_id   = max_cls;
        out.class_name = dxapp::common::get_coco_class_name(max_cls);
        out.box = {cx - w/2, cy - h/2, cx + w/2, cy + h/2};
        return true;
    }

    std::vector<AnchorYOLOResult> decoding_npu_outputs(const dxrt::TensorPtrs& outputs) const {
        std::vector<AnchorYOLOResult> detections;
        if (!npu_supported_) return detections;  // e.g. YOLOX

        auto decode_stride_npu = [&](const float* data, int stride,
                                     const auto& anchors,
                                     int gx_sz, int gy_sz) {
            for (int a = 0; a < static_cast<int>(anchors.size()); ++a) {
                int aw = anchors[a].first, ah = anchors[a].second;
                for (int gy = 0; gy < gy_sz; ++gy) {
                    for (int gx = 0; gx < gx_sz; ++gx) {
                        AnchorYOLOResult r;
                        if (decode_npu_anchor_cell(data, a, gx, gy, gx_sz, gy_sz, aw, ah, stride, r))
                            detections.push_back(std::move(r));
                    }
                }
            }
        };

        for (size_t oi = 0; oi < outputs.size(); ++oi) {
            auto shape = outputs[oi]->shape();
            int stride = std::next(anchors_by_strides_.begin(), oi)->first;
            const auto& anchors = anchors_by_strides_.at(stride);
            int gx_sz = input_width_ / stride;
            int gy_sz = input_height_ / stride;
            int num_anchors = static_cast<int>(anchors.size());
            int det_per_anchor = num_classes_ + 5;

            if (shape.size() == 5) {
                // 5D raw: [1, num_anchors, H, W, det_per_anchor] → reshape to NCHW logically
                const float* raw = static_cast<const float*>(outputs[oi]->data());
                // Repack to NCHW [1, num_anchors*det_per_anchor, H, W] for decode
                std::vector<float> nchw(num_anchors * det_per_anchor * gx_sz * gy_sz);
                for (int a = 0; a < num_anchors; ++a) {
                    for (int gy = 0; gy < gy_sz; ++gy) {
                        for (int gx = 0; gx < gx_sz; ++gx) {
                            for (int d = 0; d < det_per_anchor; ++d) {
                                int src_idx = ((a * gy_sz + gy) * gx_sz + gx) * det_per_anchor + d;
                                int dst_idx = ((a * det_per_anchor + d) * gy_sz + gy) * gx_sz + gx;
                                nchw[dst_idx] = raw[src_idx];
                            }
                        }
                    }
                }
                decode_stride_npu(nchw.data(), stride, anchors, gx_sz, gy_sz);
            } else if (shape.size() == 4 && shape[3] == num_anchors * det_per_anchor) {
                // NHWC: [1, H, W, C] → transpose to NCHW
                const float* nhwc = static_cast<const float*>(outputs[oi]->data());
                int C = static_cast<int>(shape[3]);
                std::vector<float> nchw(C * gx_sz * gy_sz);
                for (int gy = 0; gy < gy_sz; ++gy) {
                    for (int gx = 0; gx < gx_sz; ++gx) {
                        for (int c = 0; c < C; ++c) {
                            nchw[c * gy_sz * gx_sz + gy * gx_sz + gx] =
                                nhwc[gy * gx_sz * C + gx * C + c];
                        }
                    }
                }
                decode_stride_npu(nchw.data(), stride, anchors, gx_sz, gy_sz);
            } else {
                // Standard NCHW: [1, C, H, W]
                auto data = static_cast<const float*>(outputs[oi]->data());
                decode_stride_npu(data, stride, anchors, gx_sz, gy_sz);
            }
        }
        return detections;
    }

    std::vector<AnchorYOLOResult> decoding_cpu_outputs(const dxrt::TensorPtrs& outputs) const {
        // Anchor-free models (e.g. YOLOX) need grid-based box decoding
        if (!npu_supported_) {
            return decoding_cpu_outputs_anchor_free(outputs);
        }

        std::vector<AnchorYOLOResult> detections;

        auto find_best_class = [&](const float* det, float obj) -> std::pair<int, float> {
            int best_cls = -1;
            float best_conf = score_threshold_;
            for (int c = 0; c < num_classes_; ++c) {
                float conf = obj * det[5 + c];
                if (conf > best_conf) { best_conf = conf; best_cls = c; }
            }
            return {best_cls, best_conf};
        };

        for (size_t oi = 0; oi < outputs.size(); ++oi) {
            auto data = static_cast<const float*>(outputs[oi]->data());
            auto num_dets = outputs[oi]->shape()[1];
            for (int i = 0; i < num_dets; ++i) {
                const float* det = data + i * 85;
                float obj = det[4];
                if (obj < object_threshold_) continue;

                auto [max_cls, max_conf] = find_best_class(det, obj);
                if (max_cls == -1) continue;

                AnchorYOLOResult r;
                r.confidence = max_conf;
                r.class_id = max_cls;
                r.class_name = dxapp::common::get_coco_class_name(max_cls);
                r.box = {det[0] - det[2]/2, det[1] - det[3]/2,
                         det[0] + det[2]/2, det[1] + det[3]/2};
                detections.push_back(std::move(r));
            }
        }
        return detections;
    }

    /**
     * @brief Decode YOLOX ORT outputs with grid-based box decoding.
     *
     * YOLOX ORT output is [1, N, 85] where bbox values are raw offsets
     * requiring grid decode: cx = (raw_cx + grid_x) * stride,
     * cy = (raw_cy + grid_y) * stride, w = exp(raw_w) * stride,
     * h = exp(raw_h) * stride.  Objectness and class scores are
     * already sigmoid-applied.
     */
    std::vector<AnchorYOLOResult> decoding_cpu_outputs_anchor_free(
            const dxrt::TensorPtrs& outputs) const {
        std::vector<AnchorYOLOResult> detections;

        // Build grid mapping: for each detection row, store (grid_x, grid_y, stride).
        // Strides come from anchors_by_strides_ keys (sorted: 8, 16, 32).
        struct GridCell { float gx; float gy; float stride; };
        std::vector<GridCell> grid_cells;
        for (const auto& as : anchors_by_strides_) {
            int stride = as.first;
            int gx_sz = input_width_  / stride;
            int gy_sz = input_height_ / stride;
            for (int gy = 0; gy < gy_sz; ++gy) {
                for (int gx = 0; gx < gx_sz; ++gx) {
                    grid_cells.push_back({static_cast<float>(gx),
                                          static_cast<float>(gy),
                                          static_cast<float>(stride)});
                }
            }
        }

        for (const auto& output : outputs) {
            auto data = static_cast<const float*>(output->data());
            auto num_dets = static_cast<int>(output->shape()[1]);

            for (int i = 0; i < num_dets; ++i) {
                const float* det = data + i * (num_classes_ + 5);
                float obj = det[4];   // sigmoid already applied
                if (obj < object_threshold_) continue;

                auto [best_cls, best_conf] = find_best_class_score_(det, obj);
                if (best_cls == -1) continue;

                // Grid decode for YOLOX raw bbox offsets
                float cx = 0;
                float cy = 0;
                float w = 0;
                float h = 0;
                if (i < static_cast<int>(grid_cells.size())) {
                    const auto& gc = grid_cells[i];
                    cx = (det[0] + gc.gx) * gc.stride;
                    cy = (det[1] + gc.gy) * gc.stride;
                    w  = std::exp(det[2]) * gc.stride;
                    h  = std::exp(det[3]) * gc.stride;
                } else {
                    // Fallback: treat as already decoded coordinates
                    cx = det[0]; cy = det[1]; w = det[2]; h = det[3];
                }

                AnchorYOLOResult r;
                r.confidence = best_conf;
                r.class_id   = best_cls;
                r.class_name = dxapp::common::get_coco_class_name(best_cls);
                r.box = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
                detections.push_back(std::move(r));
            }
        }
        return detections;
    }

    std::vector<AnchorYOLOResult> apply_nms(const std::vector<AnchorYOLOResult>& dets) const {
        return postprocess_utils::apply_nms(dets, nms_threshold_);
    }
};

// ============================================================================
// Legacy convenience subclasses removed.
// All preset handling (anchors, NPU tensor names, thresholds) is now done by
// AnchorYOLOPostprocessor::Preset in yolo_detection_postprocessor.hpp.
//
// If you need to construct a raw AnchorYOLOPostProcess directly, pass all
// parameters to the base constructor.
// ============================================================================

#endif  // ANCHOR_DETECTION_POSTPROCESSOR_HPP
