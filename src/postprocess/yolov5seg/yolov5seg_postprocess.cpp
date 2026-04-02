#include "yolov5seg_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

#include "common_util.hpp"
#include "../../cpp_example/common/processors/postprocess_utils.hpp"

bool YOLOv5SegResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv5SegPostProcess::YOLOv5SegPostProcess(int input_w, int input_h,
                                             float obj_threshold, float score_threshold,
                                             float nms_threshold, bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    obj_threshold_ = obj_threshold;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    is_ort_configured_ = is_ort_configured;

    if (!is_ort_configured_) {
        throw std::invalid_argument(
            "ORT-OFF output postprocessing is not supported for yolov5-seg\n"
            "please dxrt build with USE_ORT=ON");
    }
}

// Default constructor
YOLOv5SegPostProcess::YOLOv5SegPostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    obj_threshold_ = 0.25f;
    score_threshold_ = 0.5f;
    nms_threshold_ = 0.45f;
    is_ort_configured_ = false;
}

// Process model outputs
std::vector<YOLOv5SegResult> YOLOv5SegPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error(
            "[DXAPP] [ER] YOLOv5SegPostProcess::postprocess - No output tensors.");
    }

    auto detections = decoding_outputs(outputs);
    detections = apply_nms(detections);
    decode_masks(outputs, detections);

    return detections;
}

// Decode detection outputs
// YOLOv5-seg output0: [1, N, 117] = [cx, cy, w, h, obj, 80 classes, 32 mask_coefs]

// Find best class index and score from the class logit region of an anchor row
static int find_best_class_row(const float* row, int num_classes, float& best_score) {
    int best_cls = 0;
    best_score = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        float s = row[5 + c];
        if (s > best_score) { best_score = s; best_cls = c; }
    }
    return best_cls;
}

std::vector<YOLOv5SegResult> YOLOv5SegPostProcess::decoding_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv5SegResult> detections;

    // Find detection tensor: shape [1, N, 117]
    const dxrt::TensorPtr* det_tensor = nullptr;
    for (const auto& t : outputs) {
        if (t->shape().size() == 3 &&
            t->shape()[2] == 5 + num_classes_ + num_mask_coefs_) {
            det_tensor = &t;
            break;
        }
    }

    if (!det_tensor) {
        std::ostringstream msg;
        msg << "[DXAPP] [ER] YOLOv5SegPostProcess - Cannot find detection tensor.\n"
            << "  Expected shape [1, N, " << (5 + num_classes_ + num_mask_coefs_) << "]\n"
            << "  Available tensors:\n";
        msg << postprocess_utils::format_tensor_shapes(outputs);
        throw std::runtime_error(msg.str());
    }

    const float* data = static_cast<const float*>((*det_tensor)->data());
    auto num_dets = (*det_tensor)->shape()[1];
    const int row_size = 5 + num_classes_ + num_mask_coefs_;  // 117

    for (int i = 0; i < num_dets; ++i) {
        const float* row = data + i * row_size;

        float obj = row[4];
        if (obj < obj_threshold_) continue;

        // Find best class via helper
        float best_score;
        int best_cls = find_best_class_row(row, num_classes_, best_score);

        float confidence = obj * best_score;
        if (confidence < score_threshold_) continue;

        // Decode box: cxcywh -> x1y1x2y2
        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        YOLOv5SegResult result;
        result.confidence = confidence;
        result.class_id = best_cls;
        result.class_name = dxapp::common::get_coco_class_name(best_cls);
        result.box = {x1, y1, x2, y2};

        // Extract mask coefficients
        result.seg_mask_coef.resize(num_mask_coefs_);
        for (int j = 0; j < num_mask_coefs_; ++j) {
            result.seg_mask_coef[j] = row[5 + num_classes_ + j];
        }

        detections.emplace_back(std::move(result));
    }

    return detections;
}

// Apply NMS
std::vector<YOLOv5SegResult> YOLOv5SegPostProcess::apply_nms(
    const std::vector<YOLOv5SegResult>& detections) const {
    if (detections.empty()) return {};

    std::vector<std::pair<float, size_t>> conf_idx;
    for (size_t i = 0; i < detections.size(); ++i) {
        conf_idx.emplace_back(detections[i].confidence, i);
    }
    std::sort(conf_idx.begin(), conf_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<YOLOv5SegResult> results;

    for (size_t i = 0; i < conf_idx.size(); ++i) {
        if (suppressed[i]) continue;
        size_t det_i = conf_idx[i].second;
        results.emplace_back(detections[det_i]);

        for (size_t j = i + 1; j < conf_idx.size(); ++j) {
            if (suppressed[j]) continue;
            size_t det_j = conf_idx[j].second;

            float x_left = std::max(detections[det_i].box[0], detections[det_j].box[0]);
            float y_top = std::max(detections[det_i].box[1], detections[det_j].box[1]);
            float x_right = std::min(detections[det_i].box[2], detections[det_j].box[2]);
            float y_bottom = std::min(detections[det_i].box[3], detections[det_j].box[3]);

            if (x_right <= x_left || y_bottom <= y_top) continue;
            float intersection = (x_right - x_left) * (y_bottom - y_top);
            float area_i = detections[det_i].area();
            float area_j = detections[det_j].area();
            float iou = intersection / (area_i + area_j - intersection);
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }
    return results;
}

// Decode segmentation masks using mask prototypes

// Compute the full-resolution binary mask for one detection and store it in det
static void decode_single_det_mask(
    YOLOv5SegResult& det,
    const float* mask_proto, int num_mask_coefs,
    int mask_area, int mask_h, int mask_w,
    int input_width, int input_height,
    float scale_w, float scale_h) {
    std::vector<float> final_mask(input_height * input_width, 0.0f);

    // Guard: coefficient vector must be present
    if (det.seg_mask_coef.size() != static_cast<size_t>(num_mask_coefs)) {
        det.mask = std::move(final_mask);
        det.mask_height = input_height;
        det.mask_width  = input_width;
        return;
    }

    // Bounding box ROI in input space
    int bx1 = std::max(0, static_cast<int>(det.box[0]));
    int by1 = std::max(0, static_cast<int>(det.box[1]));
    int bx2 = std::min(input_width,  static_cast<int>(det.box[2]));
    int by2 = std::min(input_height, static_cast<int>(det.box[3]));

    if (bx1 >= bx2 || by1 >= by2) {
        det.mask = std::move(final_mask);
        det.mask_height = input_height;
        det.mask_width  = input_width;
        return;
    }

    // ROI in prototype mask space
    int mx1 = std::max(0,      static_cast<int>(std::floor(bx1 * scale_w)));
    int my1 = std::max(0,      static_cast<int>(std::floor(by1 * scale_h)));
    int mx2 = std::min(mask_w, static_cast<int>(std::ceil(bx2 * scale_w)));
    int my2 = std::min(mask_h, static_cast<int>(std::ceil(by2 * scale_h)));

    int roi_w = mx2 - mx1;
    int roi_h = my2 - my1;
    if (roi_w <= 0 || roi_h <= 0) {
        det.mask = std::move(final_mask);
        det.mask_height = input_height;
        det.mask_width  = input_width;
        return;
    }

    // Dot product coef @ proto in ROI, then sigmoid activation
    std::vector<float> roi_mask(roi_w * roi_h, 0.0f);
    for (int c = 0; c < num_mask_coefs; ++c) {
        float coef = det.seg_mask_coef[c];
        const float* proto_plane = mask_proto + c * mask_area;
        for (int h = 0; h < roi_h; ++h) {
            const float* proto_row = proto_plane + (my1 + h) * mask_w;
            float* roi_row = roi_mask.data() + h * roi_w;
            for (int w = 0; w < roi_w; ++w)
                roi_row[w] += coef * proto_row[mx1 + w];
        }
    }
    for (float& val : roi_mask)
        val = 1.0f / (1.0f + std::exp(-val));

    // Bilinear interpolation to input space and binarize
    for (int y = by1; y < by2; ++y) {
        float src_y = y * scale_h - my1;
        int y0 = std::max(0, std::min(static_cast<int>(src_y), roi_h - 1));
        int y1 = std::min(y0 + 1, roi_h - 1);
        float dy = src_y - y0;
        float* row_ptr = &final_mask[y * input_width];

        for (int x = bx1; x < bx2; ++x) {
            float src_x = x * scale_w - mx1;
            int x0 = std::max(0, std::min(static_cast<int>(src_x), roi_w - 1));
            int x1 = std::min(x0 + 1, roi_w - 1);
            float dx = src_x - x0;

            float v00 = roi_mask[y0 * roi_w + x0];
            float v01 = roi_mask[y0 * roi_w + x1];
            float v10 = roi_mask[y1 * roi_w + x0];
            float v11 = roi_mask[y1 * roi_w + x1];
            float val = (v00 * (1.f - dx) + v01 * dx) * (1.f - dy) +
                        (v10 * (1.f - dx) + v11 * dx) * dy;
            row_ptr[x] = (val > 0.5f) ? 1.0f : 0.0f;
        }
    }

    det.mask = std::move(final_mask);
    det.mask_height = input_height;
    det.mask_width  = input_width;
}

void YOLOv5SegPostProcess::decode_masks(
    const dxrt::TensorPtrs& outputs,
    std::vector<YOLOv5SegResult>& detections) {
    if (detections.empty()) return;

    // Find mask prototype tensor: [1, 32, mask_h, mask_w]
    const float* mask_proto = nullptr;
    int mask_h = 0, mask_w = 0;
    for (const auto& t : outputs) {
        if (t->shape().size() == 4 && t->shape()[1] == num_mask_coefs_) {
            mask_proto = static_cast<const float*>(t->data());
            mask_h = static_cast<int>(t->shape()[2]);
            mask_w = static_cast<int>(t->shape()[3]);
            break;
        }
    }

    if (!mask_proto || mask_h == 0 || mask_w == 0) return;

    const int mask_area = mask_h * mask_w;
    const float scale_h = static_cast<float>(mask_h) / input_height_;
    const float scale_w = static_cast<float>(mask_w) / input_width_;

    // Delegate per-detection mask computation to helper
    for (auto& det : detections) {
        decode_single_det_mask(det, mask_proto, num_mask_coefs_,
                               mask_area, mask_h, mask_w,
                               input_width_, input_height_,
                               scale_w, scale_h);
    }
}

// Set thresholds
void YOLOv5SegPostProcess::set_thresholds(float obj_threshold, float score_threshold,
                                            float nms_threshold) {
    if (obj_threshold >= 0.0f && obj_threshold <= 1.0f) obj_threshold_ = obj_threshold;
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) score_threshold_ = score_threshold;
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) nms_threshold_ = nms_threshold;
}

// Get config info
std::string YOLOv5SegPostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv5-seg PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Objectness threshold: " << obj_threshold_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  NMS threshold: " << nms_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n"
        << "  Is ORT Configured: " << (is_ort_configured_ ? "Yes" : "No") << "\n";
    return oss.str();
}
