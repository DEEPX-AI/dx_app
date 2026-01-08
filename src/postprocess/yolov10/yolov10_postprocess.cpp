#include "yolov10_postprocess.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <sstream>

#include "common_util.hpp"

// YOLOv10Result methods implementation
float YOLOv10Result::iou(const YOLOv10Result& other) const {
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

bool YOLOv10Result::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv10PostProcess::YOLOv10PostProcess(const int input_w, const int input_h,
                                     const float score_threshold, const float nms_threshold,
                                     const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    is_ort_configured_ = is_ort_configured;

    // Initialize model-specific parameters for YOLOv10
    cpu_output_names_ = {"output0"};
    npu_output_names_ = {"/model.22/dfl/conv/Conv_output_0", "/model.22/Sigmoid_output_0"};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Default constructor
YOLOv10PostProcess::YOLOv10PostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.45;
    nms_threshold_ = 0.4;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for YOLOv10
    cpu_output_names_ = {"output0"};
    npu_output_names_ = {"/model.22/dfl/conv/Conv_output_0", "/model.22/Sigmoid_output_0"};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Process model outputs
std::vector<YOLOv10Result> YOLOv10PostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    // First align the tensors based on model configuration
    auto aligned_outputs = align_tensors(outputs);
    if (aligned_outputs.empty()) {
        int i = 0;
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv10PostProcess::postprocess - Aligned outputs are empty.\n"
            << "  Unexpected shape\n";
        for (auto& o : outputs) {
            msg << "    Output shape [" << i++ << "]: (";
            for (size_t i = 0; i < o->shape().size(); ++i) {
                msg << o->shape()[i];
                if (i != o->shape().size() - 1) msg << ", ";
            }
            msg << ")\n";
        }
        msg << "Please re-compile the model with the correct output configuration.\n";

        throw std::runtime_error(msg.str());
    }

    std::vector<YOLOv10Result> detections;
    if (is_ort_configured_) {
        detections = decoding_cpu_outputs(aligned_outputs);
    } else {
        detections = decoding_npu_outputs(aligned_outputs);
    }

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

dxrt::TensorPtrs YOLOv10PostProcess::align_tensors(const dxrt::TensorPtrs& outputs) const {
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
        if (outputs.size() == 6) {
            std::vector<dxrt::TensorPtr> cls_outputs;
            std::vector<dxrt::TensorPtr> reg_outputs;

            for (const auto& output : outputs) {
                if (output->shape().size() == 4) {
                    if (output->shape()[1] == num_classes_) {
                        cls_outputs.push_back(output);
                    } else if (output->shape()[1] == 64) {
                        reg_outputs.push_back(output);
                    }
                }
            }

            if (cls_outputs.size() == 3 && reg_outputs.size() == 3) {
                auto sort_fn = [](const dxrt::TensorPtr& a, const dxrt::TensorPtr& b) {
                    return a->shape()[2] > b->shape()[2];
                };
                std::sort(cls_outputs.begin(), cls_outputs.end(), sort_fn);
                std::sort(reg_outputs.begin(), reg_outputs.end(), sort_fn);

                for (size_t i = 0; i < 3; ++i) {
                    aligned.push_back(reg_outputs[i]);
                    aligned.push_back(cls_outputs[i]);
                }
                return aligned;
            }
        }
        return aligned;
    }
}

// Decode model outputs to detection results
std::vector<YOLOv10Result> YOLOv10PostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv10Result> detections;
    if (outputs.size() == 6) {
        for (size_t i = 0; i < 3; ++i) {
            const auto& reg_tensor = outputs[2 * i];
            const auto& cls_tensor = outputs[2 * i + 1];

            const float* reg_data = static_cast<const float*>(reg_tensor->data());
            const float* cls_data = static_cast<const float*>(cls_tensor->data());

            int H = cls_tensor->shape()[2];
            int W = cls_tensor->shape()[3];
            int stride = input_width_ / W;
            int num_grid = H * W;

            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int spatial_offset = h * W + w;

                    int max_cls = -1;
                    float max_cls_conf = score_threshold_;

                    for (int c = 0; c < num_classes_; ++c) {
                        float conf = cls_data[c * num_grid + spatial_offset];
                        if (conf > max_cls_conf) {
                            max_cls_conf = conf;
                            max_cls = c;
                        }
                    }

                    if (max_cls != -1) {
                        float dist[4];
                        for (int k = 0; k < 4; ++k) {
                            float exp_sum = 0.0f;
                            float weighted_sum = 0.0f;
                            float max_val = -std::numeric_limits<float>::infinity();

                            for (int d = 0; d < 16; ++d) {
                                float val = reg_data[(k * 16 + d) * num_grid + spatial_offset];
                                if (val > max_val) max_val = val;
                            }

                            for (int d = 0; d < 16; ++d) {
                                float val = reg_data[(k * 16 + d) * num_grid + spatial_offset];
                                float e = std::exp(val - max_val);
                                exp_sum += e;
                                weighted_sum += e * d;
                            }
                            dist[k] = weighted_sum / exp_sum;
                        }

                        float anchor_x = w + 0.5f;
                        float anchor_y = h + 0.5f;

                        float x1 = (anchor_x - dist[0]) * stride;
                        float y1 = (anchor_y - dist[1]) * stride;
                        float x2 = (anchor_x + dist[2]) * stride;
                        float y2 = (anchor_y + dist[3]) * stride;

                        YOLOv10Result result;
                        result.confidence = max_cls_conf;
                        result.class_id = max_cls;
                        result.class_name = dxapp::common::get_coco_class_name(max_cls);
                        result.box = {x1, y1, x2, y2};
                        detections.push_back(result);
                    }
                }
            }
        }
    }
    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOv10Result> YOLOv10PostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv10Result> detections;

    // Assume the first tensor has shape [1, 300, 6] with layout:
    // [x1, y1, x2, y2, score, class_id], same as Python yolov10_sync.
    if (outputs.empty()) {
        return detections;
    }

    const auto& tensor = outputs[0];
    const float* data = static_cast<const float*>(tensor->data());
    const auto& shape = tensor->shape();

    if (shape.size() != 3) {
        return detections;
    }

    const int num_dets = static_cast<int>(shape[1]);  // e.g., 300
    const int stride = static_cast<int>(shape[2]);     // e.g., 6

    for (int i = 0; i < num_dets; ++i) {
        const float* det = data + i * stride;

        float score = det[4];
        if (score < score_threshold_) {
            continue;
        }

        int class_id = static_cast<int>(det[5]);
        if (class_id < 0 || class_id >= num_classes_) {
            continue;
        }

        YOLOv10Result result;
        result.confidence = score;
        result.class_id = class_id;
        result.class_name = dxapp::common::get_coco_class_name(class_id);
        result.box = {det[0], det[1], det[2], det[3]};

        detections.push_back(result);
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<YOLOv10Result> YOLOv10PostProcess::apply_nms(
    const std::vector<YOLOv10Result>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<YOLOv10Result> sorted_detections = detections;
    std::sort(
        sorted_detections.begin(), sorted_detections.end(),
        [](const YOLOv10Result& a, const YOLOv10Result& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOv10Result> result;

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
void YOLOv10PostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
std::string YOLOv10PostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv10 PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
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
