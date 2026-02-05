#include "yolov26seg_postprocess.h"

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

bool YOLOv26SegResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv26SegPostProcess::YOLOv26SegPostProcess(const int input_w, const int input_h,
                                             const float score_threshold,
                                             const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    is_ort_configured_ = is_ort_configured;

    if (!is_ort_configured_) {
        throw std::invalid_argument(
            "ORT-OFF output postprocessing is not supported for yolov26-seg. "
            "Please build dxrt with USE_ORT=ON.");
    }

    // NMS-free YOLOv26-seg (ORT) output layout:
    //   output0: FLOAT, [1, 300, 38]       -> bbox(4) + score + class_id + seg_coef
    //   output1: FLOAT, [1, 32, 160, 160]  -> mask prototypes

    cpu_output_names_ = {"output0", "output1"};
}

// Default constructor
YOLOv26SegPostProcess::YOLOv26SegPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.5f;
    is_ort_configured_ = false;

    cpu_output_names_ = {"output0", "output1"};
}

// Process model outputs (NMS-free)
std::vector<YOLOv26SegResult> YOLOv26SegPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (!is_ort_configured_) {
        throw std::runtime_error("YOLOv26SegPostProcess currently supports only ORT inference mode.");
    }

    if (outputs.size() != 2) {
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv26SegPostProcess::postprocess - Unexpected number of outputs: "
            << outputs.size() << " (expected 2).";
        throw std::runtime_error(msg.str());
    }

    // 1. Decode detections (bbox + score + class_id + mask coefficients) with score thresholding
    std::vector<YOLOv26SegResult> detections = decoding_cpu_outputs(outputs);

    // 2. Decode segmentation masks using prototypes (if detections exist)
    decoding_mask_cpu_outputs(outputs, detections);

    return detections;
}

void YOLOv26SegPostProcess::decoding_mask_cpu_outputs(const dxrt::TensorPtrs& outputs,
                                                      std::vector<YOLOv26SegResult>& detections) {
    if (detections.empty()) return;

    /**
     * @note NMS-free YOLOv26-seg output format:
     * output0: [1, 300, 38] - bbox (4) + score + class_id + seg_coef
     * output1: [1, 32, 160, 160] - mask prototypes   (used here)
     */
    const auto& mask_tensor = outputs[1];
    const auto& shape = mask_tensor->shape();
    if (shape.size() != 4 || shape[0] != 1) {
        throw std::runtime_error(
            "Unexpected mask output shape for YOLOv26SegPostProcess (expected [1,C,H,W]).");
    }

    const int mask_height = static_cast<int>(shape[2]);
    const int mask_width = static_cast<int>(shape[3]);

    const float* mask_output = static_cast<const float*>(mask_tensor->data());
    if (mask_output) {
        auto masks = process_segmentation_masks(mask_output, detections, mask_height, mask_width);
        for (size_t i = 0; i < detections.size() && i < masks.size(); ++i) {
            detections[i].mask = std::move(masks[i]);
            detections[i].mask_height = input_height_;
            detections[i].mask_width = input_width_;
        }
    }
}

// Decode model outputs to detection results (NMS-free)
std::vector<YOLOv26SegResult> YOLOv26SegPostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    const auto& det_output = outputs[0];

    // Expected detection output shape: [1, 300, 38]
    const auto& shape = det_output->shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error(
            "Unexpected detection output rank or batch size for YOLOv26SegPostProcess");
    }

    const int num_dets = static_cast<int>(shape[1]);
    const int vec_size = static_cast<int>(shape[2]);

    if (vec_size < 6) {
        throw std::runtime_error(
            "Detection vector size must be at least 6 (x1,y1,x2,y2,score,class_id).");
    }

    const float* data = static_cast<const float*>(det_output->data());

    std::vector<YOLOv26SegResult> detections;
    detections.reserve(num_dets);

    for (int i = 0; i < num_dets; ++i) {
        const float* det = data + i * vec_size;

        const float x1 = det[0];
        const float y1 = det[1];
        const float x2 = det[2];
        const float y2 = det[3];
        const float score = det[4];

        if (score < score_threshold_) {
            continue;
        }

        YOLOv26SegResult result;
        result.box = {x1, y1, x2, y2};
        result.confidence = score;

        // Class id is provided directly; class name can be resolved externally if needed
        result.class_id = static_cast<int>(det[5]);
        result.class_name = dxapp::common::get_coco_class_name(result.class_id);

        // Remaining elements are mask coefficients
        const int num_coefs = vec_size - 6;
        result.seg_mask_coef.resize(num_coefs);
        for (int c = 0; c < num_coefs; ++c) {
            result.seg_mask_coef[c] = det[6 + c];
        }

        detections.emplace_back(std::move(result));
    }

    return detections;
}


// Set thresholds (score only, NMS is not used)
void YOLOv26SegPostProcess::set_thresholds(float score_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
}

// Get configuration information
std::string YOLOv26SegPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv26Seg PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  Is Ort Configured: " << (is_ort_configured_ ? "Yes" : "No") << "\n";

    for (auto& cpu_output_name : cpu_output_names_) {
        oss << "  CPU output name: " << cpu_output_name << "\n";
    }

    return oss.str();
}

// Note: align_tensors is no longer required for the new
// NMS-free YOLOv26Seg model where output order and shapes
// are fixed (output0: detections, output1: prototypes).

// Process segmentation masks using optimized ROI-based approach
std::vector<std::vector<float>> YOLOv26SegPostProcess::process_segmentation_masks(
    const float* mask_output, const std::vector<YOLOv26SegResult>& detections, int mask_height,
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

// Note: scale_masks and crop_masks helper functions from the previous
// implementation have been removed as separate steps. The current
// process_segmentation_masks implementation directly generates binary
// masks at the model input resolution and applies bounding-box
// cropping, so no further scaling/cropping is required here.
