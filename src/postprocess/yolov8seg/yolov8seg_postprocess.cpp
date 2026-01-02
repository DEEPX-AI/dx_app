#include "yolov8seg_postprocess.h"

#include <dxrt/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>

#include "common_util.hpp"

bool YOLOv8SegResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv8SegPostProcess::YOLOv8SegPostProcess(const int input_w, const int input_h,
                                             const float score_threshold, const float nms_threshold,
                                             const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    is_ort_configured_ = is_ort_configured;

    if (!is_ort_configured_) {
        throw std::invalid_argument(
            "ORT-OFF output postprocessing is not supported for yolov8-seg\n"
            "please dxrt build with USE_ORT=ON");
    }

    // YOLOv8-seg (ORT) output layout:
    //   output0: FLOAT, [1, 116, 8400]  -> bbox(4) + classes(80) + seg_coef(32)
    //   output1: FLOAT, [1, 32, 160, 160] -> mask prototypes

    // Initialize model-specific parameters for YOLOv8-seg
    cpu_output_names_ = {"output0", "output1"};
    npu_output_names_ = {};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Default constructor
YOLOv8SegPostProcess::YOLOv8SegPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.5f;  // Increased from 0.45f for stricter filtering
    nms_threshold_ = 0.45f;   // Increased from 0.4f for stricter NMS
    is_ort_configured_ = false;

    // YOLOv8-seg (ORT) output layout:
    //   output0: FLOAT, [1, 116, 8400]  -> bbox(4) + classes(80) + seg_coef(32)
    //   output1: FLOAT, [1, 32, 160, 160] -> mask prototypes

    // Initialize model-specific parameters for YOLOv8-seg
    cpu_output_names_ = {"output0", "output1"};
    npu_output_names_ = {};
    anchors_by_strides_ = {{8, {}}, {16, {}}, {32, {}}};
}

// Process model outputs
std::vector<YOLOv8SegResult> YOLOv8SegPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    dxrt::TensorPtrs aligned_outputs;
    if (!is_ort_configured_)
        aligned_outputs = align_tensors(outputs);
    else
        aligned_outputs = outputs;
    if (aligned_outputs.empty()) {
        int i = 0;
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv8SegPostProcess::postprocess - Aligned outputs are empty.\n"
            << "  Unexpected shape\n";
        for (auto& o : outputs) {
            msg << "    Output shape [" << i++ << "]: (";
            for (size_t i = 0; i < o->shape().size(); ++i) {
                msg << o->shape()[i];
                if (i != o->shape().size() - 1) msg << ", ";
            }
            msg << ")\n";
        }
        msg << ", Expected (1, 116, 8400) and (1, 32, 160, 160).\n"
            << "Please re-compile the model with the correct output configuration.\n";

        throw std::runtime_error(msg.str());  // 안전한 종료: 상위로 에러 전달
    }

    std::vector<YOLOv8SegResult> detections;
    detections = decoding_cpu_outputs(aligned_outputs);
    // Apply Non-Maximum Suppression (mask will be included in NMS process)
    detections = apply_nms(detections);
    /////////////////////////////////////////////////////////////////////////// OK

    // Process segmentation masks After NMS to maintain index alignment
    decoding_mask_cpu_outputs(aligned_outputs, detections);

    return detections;
}

void YOLOv8SegPostProcess::decoding_mask_cpu_outputs(const dxrt::TensorPtrs& outputs,
                                                      std::vector<YOLOv8SegResult>& detections) {
    // std::vector<YOLOv8SegResult> results;
    /**
     * @note YOLOv8-seg has different output format:
     * output0: [1, 116, 8400] - contains bbox (4) + classes (80) + seg_coef (32)
     * output1: [1, 32, 160, 160] - segmentation masks   (*** Used in this field)
     */
    const float* mask_output = static_cast<const float*>(outputs[1]->data());
    int mask_height = 160, mask_width = 160;
    if (mask_output && !detections.empty()) {
        auto masks = process_segmentation_masks(mask_output, detections, mask_height, mask_width);
        for (size_t i = 0; i < detections.size() && i < masks.size(); ++i) {
            detections[i].mask = std::move(masks[i]);
            detections[i].mask_height = input_height_;
            detections[i].mask_width = input_width_;
        }
    }
}

// Decode model outputs to detection results
std::vector<YOLOv8SegResult> YOLOv8SegPostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv8SegResult> detections;
    /**
     * @note YOLOv8-seg has different output format:
     * output0: [1, 116, 8400] - contains bbox (4) + classes (80) + seg_coef (32) (Used in this
     * field) output1: [1, 32, 160, 160] - segmentation masks
     */
    const float* bbox_output = static_cast<const float*>(outputs[0]->data());
    auto num_dets = outputs[0]->shape()[2];  // 8400

    // Optimization: Transpose the loop to access memory sequentially for class scores
    // This significantly improves cache locality as the tensor shape is [1, 116, 8400]
    std::vector<float> max_scores(num_dets, 0.0f);
    std::vector<int> best_classes(num_dets, -1);

    // 1. Find best class and score for each anchor
    // Iterate channels first, then anchors to access memory sequentially
    for (int c = 0; c < num_classes_; ++c) {
        const float* class_scores = bbox_output + (4 + c) * num_dets;
        for (int i = 0; i < num_dets; ++i) {
            float score = class_scores[i];
            if (score > max_scores[i]) {
                max_scores[i] = score;
                best_classes[i] = c;
            }
        }
    }

    // 2. Filter by threshold and extract box/mask info
    for (int i = 0; i < num_dets; ++i) {
        if (max_scores[i] < score_threshold_) {
            continue;
        }

        // Extract coordinates (xywh format like Python)
        // Strided access here, but only for valid detections (sparse)
        float cx = bbox_output[i];
        float cy = bbox_output[i + num_dets];
        float w = bbox_output[i + 2 * num_dets];
        float h = bbox_output[i + 3 * num_dets];

        // Convert to xyxy like Python ops.xywh2xyxy
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        YOLOv8SegResult result;
        result.confidence = max_scores[i];
        result.class_id = best_classes[i];
        result.class_name = dxapp::common::get_coco_class_name(result.class_id);
        result.box.resize(4);
        result.box[0] = x1;
        result.box[1] = y1;
        result.box[2] = x2;
        result.box[3] = y2;

        // Extract seg coefficients like Python x[..., 84:84+32]
        result.seg_mask_coef.resize(32);
        const float* coefs = bbox_output + 84 * num_dets;
        for (int j = 0; j < 32; ++j) {
            result.seg_mask_coef[j] = coefs[j * num_dets + i];
        }

        detections.emplace_back(std::move(result));
    }

    return detections;
}

// Apply Non-Maximum Suppression - simple version like Python torchvision.ops.nms
std::vector<YOLOv8SegResult> YOLOv8SegPostProcess::apply_nms(
    const std::vector<YOLOv8SegResult>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort by confidence (like Python)
    std::vector<std::pair<float, size_t>> conf_idx_pairs;
    for (size_t i = 0; i < detections.size(); ++i) {
        conf_idx_pairs.emplace_back(detections[i].confidence, i);
    }

    std::sort(conf_idx_pairs.begin(), conf_idx_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<YOLOv8SegResult> results;

    for (size_t i = 0; i < conf_idx_pairs.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        size_t det_i = conf_idx_pairs[i].second;
        results.emplace_back(detections[det_i]);

        // Calculate IoU with remaining boxes
        for (size_t j = i + 1; j < conf_idx_pairs.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            size_t det_j = conf_idx_pairs[j].second;

            // Calculate IoU
            float x1_i = detections[det_i].box[0];
            float y1_i = detections[det_i].box[1];
            float x2_i = detections[det_i].box[2];
            float y2_i = detections[det_i].box[3];

            float x1_j = detections[det_j].box[0];
            float y1_j = detections[det_j].box[1];
            float x2_j = detections[det_j].box[2];
            float y2_j = detections[det_j].box[3];

            float x_left = std::max(x1_i, x1_j);
            float y_top = std::max(y1_i, y1_j);
            float x_right = std::min(x2_i, x2_j);
            float y_bottom = std::min(y2_i, y2_j);

            if (x_right > x_left && y_bottom > y_top) {
                float intersection = (x_right - x_left) * (y_bottom - y_top);
                float area_i = (x2_i - x1_i) * (y2_i - y1_i);
                float area_j = (x2_j - x1_j) * (y2_j - y1_j);
                float iou = intersection / (area_i + area_j - intersection);

                if (iou > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return results;
}

// Set thresholds
void YOLOv8SegPostProcess::set_thresholds(float score_threshold, float nms_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information56
std::string YOLOv8SegPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv8n PostProcess Configuration:\n"
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

dxrt::TensorPtrs YOLOv8SegPostProcess::align_tensors(const dxrt::TensorPtrs& outputs) const {
    dxrt::TensorPtrs aligned;

    if (is_ort_configured_) {
        // YOLOv8-seg ORT outputs should be aligned as:
        // aligned[0]: [1, 116, 8400] - bbox + classes + seg_coef (detection output)
        // aligned[1]: [1, 32, 160, 160] - segmentation masks (mask output)

        dxrt::TensorPtr detection_output = nullptr;
        dxrt::TensorPtr mask_output = nullptr;

        for (const auto& output : outputs) {
            if (output->shape().size() == 3 && output->shape()[1] == 116) {
                // This is the detection output (bbox + classes + seg_coef)
                detection_output = output;
            } else if (output->shape().size() == 4 && output->shape()[1] == 32) {
                // This is the mask output
                mask_output = output;
            }
        }

        // Ensure correct order: detection first, then mask
        if (detection_output) {
            aligned.push_back(detection_output);
        }
        if (mask_output) {
            aligned.push_back(mask_output);
        }

        return aligned;
    } else {
        // YOLOv8 NPU outputs for segmentation would be similar but may have different tensor names
        for (const auto& output : outputs) {
            if (output->shape().size() == 4 && output->shape()[2] == 4) {
                // This is the boxes output
                aligned.push_back(output);
            } else if (output->shape().size() == 3 && output->shape()[1] == num_classes_) {
                // This is the scores output
                aligned.push_back(output);
            }
        }
        return aligned;
    }
}

// Process segmentation masks using optimized ROI-based approach
std::vector<std::vector<float>> YOLOv8SegPostProcess::process_segmentation_masks(
    const float* mask_output, const std::vector<YOLOv8SegResult>& detections, int mask_height,
    int mask_width) const {
    std::vector<std::vector<float>> result_masks;
    result_masks.reserve(detections.size());

    if (!mask_output || detections.empty()) {
        return result_masks;
    }

    const int num_prototypes = 32;
    const int input_h = input_height_;
    const int input_w = input_width_;
    const int mask_area = mask_height * mask_width;

    // Pre-calculate scale factors
    const float scale_h = static_cast<float>(mask_height) / input_h;
    const float scale_w = static_cast<float>(mask_width) / input_w;

    for (const auto& detection : detections) {
        // Initialize full mask with zeros
        std::vector<float> final_mask(input_h * input_w, 0.0f);

        if (detection.seg_mask_coef.size() != num_prototypes) {
            result_masks.emplace_back(std::move(final_mask));
            continue;
        }

        // 1. Determine Bounding Box in Input Image (Target ROI)
        int x1 = std::max(0, (int)detection.box[0]);
        int y1 = std::max(0, (int)detection.box[1]);
        int x2 = std::min(input_w, (int)detection.box[2]);
        int y2 = std::min(input_h, (int)detection.box[3]);

        if (x1 >= x2 || y1 >= y2) {
            result_masks.emplace_back(std::move(final_mask));
            continue;
        }

        // 2. Determine ROI in Mask Prototype Space (Source ROI)
        // Map the bounding box to the mask prototype dimensions (160x160)
        // Use floor/ceil to ensure we cover the necessary source pixels for interpolation
        int mx1 = std::max(0, static_cast<int>(std::floor(x1 * scale_w)));
        int my1 = std::max(0, static_cast<int>(std::floor(y1 * scale_h)));
        int mx2 = std::min(mask_width, static_cast<int>(std::ceil(x2 * scale_w)));
        int my2 = std::min(mask_height, static_cast<int>(std::ceil(y2 * scale_h)));

        int roi_w = mx2 - mx1;
        int roi_h = my2 - my1;

        if (roi_w <= 0 || roi_h <= 0) {
            result_masks.emplace_back(std::move(final_mask));
            continue;
        }

        // 3. Compute Mask Values ONLY for the ROI
        // This avoids computing dot products for the entire 160x160 grid
        std::vector<float> roi_mask(roi_w * roi_h, 0.0f);

        // Optimization: Iterate prototypes outer loop to improve cache locality for mask_output
        for (int c = 0; c < num_prototypes; ++c) {
            float coef = detection.seg_mask_coef[c];
            const float* proto_plane = mask_output + c * mask_area;

            for (int h = 0; h < roi_h; ++h) {
                int global_h = my1 + h;
                const float* proto_row = proto_plane + global_h * mask_width;
                float* roi_row = roi_mask.data() + h * roi_w;

                for (int w = 0; w < roi_w; ++w) {
                    int global_w = mx1 + w;
                    roi_row[w] += coef * proto_row[global_w];
                }
            }
        }

        // Apply sigmoid to the ROI mask
        for (float& val : roi_mask) {
            val = 1.0f / (1.0f + std::exp(-val));
        }

        // 4. Resize ROI to Bounding Box and Place in Final Mask
        // We only iterate over the bounding box area in the final mask
        for (int y = y1; y < y2; ++y) {
            // Map to ROI coordinates
            float src_y = y * scale_h - my1;
            int y0 = static_cast<int>(src_y);
            int y1_idx = std::min(y0 + 1, roi_h - 1);
            float dy = src_y - y0;
            
            // Clamp y0 to be safe
            y0 = std::max(0, std::min(y0, roi_h - 1));

            // Pointer to the row in final mask
            float* row_ptr = &final_mask[y * input_w];

            for (int x = x1; x < x2; ++x) {
                float src_x = x * scale_w - mx1;
                int x0 = static_cast<int>(src_x);
                int x1_idx = std::min(x0 + 1, roi_w - 1);
                float dx = src_x - x0;

                // Clamp x0 to be safe
                x0 = std::max(0, std::min(x0, roi_w - 1));

                // Bilinear interpolation within ROI
                float v00 = roi_mask[y0 * roi_w + x0];
                float v01 = roi_mask[y0 * roi_w + x1_idx];
                float v10 = roi_mask[y1_idx * roi_w + x0];
                float v11 = roi_mask[y1_idx * roi_w + x1_idx];

                float val = (v00 * (1.0f - dx) + v01 * dx) * (1.0f - dy) + 
                            (v10 * (1.0f - dx) + v11 * dx) * dy;

                // Apply threshold (binarize)
                row_ptr[x] = (val > 0.5f) ? 1.0f : 0.0f;
            }
        }

        result_masks.emplace_back(std::move(final_mask));
    }

    return result_masks;
}

// Scale masks from original size to target size
std::vector<std::vector<float>> YOLOv8SegPostProcess::scale_masks(
    std::vector<std::vector<float>>&& masks, int target_height, int target_width, int orig_height,
    int orig_width) const {
    // If no scaling needed, return the masks as-is without copying
    if (target_height == orig_height && target_width == orig_width) {
        return std::move(masks);
    }

    std::vector<std::vector<float>> scaled_masks;
    scaled_masks.reserve(masks.size());

    const float scale_h = static_cast<float>(target_height) / orig_height;
    const float scale_w = static_cast<float>(target_width) / orig_width;
    const int target_size = target_height * target_width;

    for (auto& mask : masks) {
        std::vector<float> scaled_mask;
        scaled_mask.reserve(target_size);
        scaled_mask.resize(target_size);

        // Simple bilinear interpolation
        for (int th = 0; th < target_height; ++th) {
            const float orig_h = th / scale_h;
            const int h0 = static_cast<int>(orig_h);
            const int h1 = std::min(h0 + 1, orig_height - 1);
            const float dh = orig_h - h0;

            for (int tw = 0; tw < target_width; ++tw) {
                const float orig_w = tw / scale_w;
                const int w0 = static_cast<int>(orig_w);
                const int w1 = std::min(w0 + 1, orig_width - 1);
                const float dw = orig_w - w0;

                // Bounds checking
                const int safe_h0 = std::max(0, std::min(h0, orig_height - 1));
                const int safe_w0 = std::max(0, std::min(w0, orig_width - 1));
                const int safe_h1 = std::max(0, std::min(h1, orig_height - 1));
                const int safe_w1 = std::max(0, std::min(w1, orig_width - 1));

                const float val00 = mask[safe_h0 * orig_width + safe_w0];
                const float val01 = mask[safe_h0 * orig_width + safe_w1];
                const float val10 = mask[safe_h1 * orig_width + safe_w0];
                const float val11 = mask[safe_h1 * orig_width + safe_w1];

                const float val0 = val00 * (1.0f - dw) + val01 * dw;
                const float val1 = val10 * (1.0f - dw) + val11 * dw;
                const float interpolated_val = val0 * (1.0f - dh) + val1 * dh;

                scaled_mask[th * target_width + tw] = interpolated_val;
            }
        }

        scaled_masks.emplace_back(std::move(scaled_mask));
    }

    return scaled_masks;
}

// Crop masks to bounding box regions
std::vector<std::vector<float>> YOLOv8SegPostProcess::crop_masks(
    std::vector<std::vector<float>>&& masks,
    const std::vector<YOLOv8SegResult>& detections) const {
    if (masks.size() != detections.size()) {
        return std::move(masks);  // Size mismatch, return original masks
    }

    // Work directly on the input masks to avoid copying
    for (size_t i = 0; i < masks.size(); ++i) {
        auto& mask = masks[i];  // Work directly on the original mask
        const auto& detection = detections[i];

        if (detection.box.size() < 4) {
            continue;  // Invalid box, keep original mask
        }

        // Get bounding box coordinates (normalized to input size)
        const int x1 = static_cast<int>(std::max(0.0f, detection.box[0]));
        const int y1 = static_cast<int>(std::max(0.0f, detection.box[1]));
        const int x2 =
            static_cast<int>(std::min(static_cast<float>(input_width_), detection.box[2]));
        const int y2 =
            static_cast<int>(std::min(static_cast<float>(input_height_), detection.box[3]));

        // Apply cropping and thresholding in a single pass
        for (int h = 0; h < input_height_; ++h) {
            const int row_offset = h * input_width_;
            const bool in_y_range = (h >= y1 && h < y2);

            for (int w = 0; w < input_width_; ++w) {
                const int idx = row_offset + w;

                if (!in_y_range || w < x1 || w >= x2) {
                    // Outside bounding box - set to 0
                    mask[idx] = 0.0f;
                } else {
                    // Inside bounding box - apply threshold to create binary mask
                    mask[idx] = (mask[idx] > 0.5f) ? 1.0f : 0.0f;
                }
            }
        }
    }

    return std::move(masks);
}
