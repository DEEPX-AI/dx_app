#include "yolox_postprocess.h"

#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

#include "common_util.hpp"

// YOLOXResult methods implementation
float YOLOXResult::iou(const YOLOXResult& other) const {
    // Calculate intersection coordinates
    float x_left = std::max(box[0], other.box[0]);
    float y_top = std::max(box[1], other.box[1]);
    float x_right = std::min(box[2], other.box[2]);
    float y_bottom = std::min(box[3], other.box[3]);

    // Check if there is intersection
    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float union_area = area() + other.area() - intersection_area;

    return intersection_area / union_area;
}
bool YOLOXResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOXPostProcess::YOLOXPostProcess(const int input_w, const int input_h, const float obj_threshold,
                                   const float score_threshold, const float nms_threshold,
                                   const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    object_threshold_ = obj_threshold;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    is_ort_configured_ = is_ort_configured;

    // Initialize model-specific parameters for YOLOXs
    cpu_output_names_ = {"output"};
    npu_output_names_ = {};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Default constructor
YOLOXPostProcess::YOLOXPostProcess() {
    input_width_ = 512;
    input_height_ = 512;
    object_threshold_ = 0.25f;
    score_threshold_ = 0.3f;
    nms_threshold_ = 0.45f;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for YOLOXs
    cpu_output_names_ = {"output"};
    npu_output_names_ = {};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Process model outputs
std::vector<YOLOXResult> YOLOXPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    // First align the tensors based on model configuration
    auto aligned_outputs = align_tensors(outputs);

    std::vector<YOLOXResult> detections;
    if (is_ort_configured_) {
        detections = decoding_cpu_outputs(aligned_outputs);
    } else {
        detections = decoding_npu_outputs(aligned_outputs);
    }

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Align output tensors based on model configuration
dxrt::TensorPtrs YOLOXPostProcess::align_tensors(const dxrt::TensorPtrs& outputs) const {
    dxrt::TensorPtrs aligned;

    if (is_ort_configured_) {
        for (const auto& output : outputs) {
            if (output->shape().size() == 3) {
                aligned.push_back(output);
                break;
            }
        }
        return aligned;  // ORT inference does not require reordering
    } else {
        // Align outputs based on anchors_by_strides
        for (const auto& as : anchors_by_strides_) {
            for (const auto& output : outputs) {
                if (output->shape().size() == 4 && output->shape()[2] == input_width_ / as.first &&
                    output->shape()[3] == input_height_ / as.first &&
                    output->shape()[1] ==
                        static_cast<int64_t>((num_classes_ + 5) * as.second.size())) {
                    aligned.push_back(output);
                    break;
                }
            }
        }
    }

    if (aligned.empty()) {
        std::cerr << "[DXAPP] [ERROR] Failed to align output tensors based on "
                     "anchors_by_strides."
                  << std::endl;
        return outputs;  // Fallback to original outputs if alignment fails
    }

    return aligned;
}

// Decode model outputs to detection results
std::vector<YOLOXResult> YOLOXPostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOXResult> detections;
    (void)outputs;
    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOXResult> YOLOXPostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOXResult> detections;

    // YOLOXs typically has 1 output tensor
    // output tensor contains: [batch, number of detections, attribute_size]
    // Where attribute_size = [x, y, w, h, obj_conf, cls_conf_0, ..., cls_conf_(num_classes-1)]
    //
    // The x/y/w/h may be:
    //   (a) Raw grid-relative values (need stride decode), or
    //   (b) Already decoded pixel coordinates.
    // Detect which by checking the value range of the first dimension.
    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const float* output = static_cast<const float*>(outputs[output_idx]->data());
        auto num_dets = outputs[output_idx]->shape()[1];
        const int attribute_size = static_cast<int>(outputs[output_idx]->shape()[2]);

        // --- Auto-detect whether grid decode is needed ---
        // Sample up to 200 x-values; if max(|x|) < 20 they are grid-relative.
        bool needs_grid_decode = false;
        {
            float max_abs_x = 0.0f;
            int sample_count = std::min(static_cast<int>(num_dets), 200);
            int step = std::max(1, static_cast<int>(num_dets) / sample_count);
            for (int i = 0; i < num_dets && sample_count > 0; i += step, --sample_count) {
                float ax = std::abs(output[i * attribute_size]);
                if (ax > max_abs_x) max_abs_x = ax;
            }
            needs_grid_decode = (max_abs_x < 20.0f);
        }

        // --- Build grid if needed ---
        // YOLOX uses strides [8, 16, 32]. Feature map sizes: input_dim / stride.
        struct GridEntry { float gx; float gy; float stride; };
        std::vector<GridEntry> grid;
        if (needs_grid_decode) {
            const int strides[] = {8, 16, 32};
            int total = 0;
            for (int s : strides) {
                int gh = input_height_ / s;
                int gw = input_width_ / s;
                total += gh * gw;
            }
            grid.reserve(total);
            for (int s : strides) {
                int gh = input_height_ / s;
                int gw = input_width_ / s;
                for (int y = 0; y < gh; ++y) {
                    for (int x = 0; x < gw; ++x) {
                        grid.push_back({static_cast<float>(x),
                                        static_cast<float>(y),
                                        static_cast<float>(s)});
                    }
                }
            }
        }

        for (int i = 0; i < num_dets; ++i) {
            const float* det = output + i * attribute_size;
            auto objectness_score = det[4];
            if (objectness_score < object_threshold_) {
                continue;
            }
            // Find the class with highest confidence
            int max_cls = -1;
            float max_cls_conf = score_threshold_;
            for (int cls = 0; cls < num_classes_; ++cls) {
                float class_conf = objectness_score * det[5 + cls];
                if (class_conf > max_cls_conf) {
                    max_cls_conf = class_conf;
                    max_cls = cls;
                }
            }
            if (max_cls == -1) {
                continue;
            }

            float cx, cy, bw, bh;
            if (needs_grid_decode && i < static_cast<int>(grid.size())) {
                float s = grid[i].stride;
                cx = (det[0] + grid[i].gx) * s;
                cy = (det[1] + grid[i].gy) * s;
                bw = std::exp(det[2]) * s;
                bh = std::exp(det[3]) * s;
            } else {
                cx = det[0];
                cy = det[1];
                bw = det[2];
                bh = det[3];
            }

            YOLOXResult result;
            result.confidence = max_cls_conf;
            result.class_id = max_cls;
            result.class_name = dxapp::common::get_coco_class_name(max_cls);

            // Convert center coordinates to corner coordinates
            result.box.emplace_back(cx - bw / 2.0f);  // x1
            result.box.emplace_back(cy - bh / 2.0f);  // y1
            result.box.emplace_back(cx + bw / 2.0f);  // x2
            result.box.emplace_back(cy + bh / 2.0f);  // y2

            detections.push_back(result);
        }
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<YOLOXResult> YOLOXPostProcess::apply_nms(
    const std::vector<YOLOXResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<YOLOXResult> sorted_detections = detections;
    std::sort(
        sorted_detections.begin(), sorted_detections.end(),
        [](const YOLOXResult& a, const YOLOXResult& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOXResult> result;

    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        result.push_back(sorted_detections[i]);

        // Suppress overlapping detections
        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (!suppressed[j]) {
                float iou_value = sorted_detections[i].iou(sorted_detections[j]);
                if (iou_value > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

// Set thresholds
void YOLOXPostProcess::set_thresholds(float obj_threshold, float score_threshold,
                                      float nms_threshold) {
    if (obj_threshold >= 0.0f && obj_threshold <= 1.0f) {
        object_threshold_ = obj_threshold;
    }
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
std::string YOLOXPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOXs PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Object threshold: " << object_threshold_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  NMS threshold: " << nms_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n"
        << "  Is Ort Configured: " << (is_ort_configured_ ? "Yes" : "No") << "\n";

    for (auto& as : anchors_by_strides_) {
        oss << "  Stride: " << as.first << " Anchors: ";
        for (auto& a : as.second) {
            oss << a.first << ", " << a.second << " | ";
        }
        oss << "\n";
    }
    for (auto& cpu_output_name : cpu_output_names_) {
        oss << "  CPU output name: " << cpu_output_name << "\n";
    }
    for (auto& npu_output_name : npu_output_names_) {
        oss << "  NPU output name: " << npu_output_name << "\n";
    }

    return oss.str();
}
