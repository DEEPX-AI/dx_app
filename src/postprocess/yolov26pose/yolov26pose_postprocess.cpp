#include "yolov26pose_postprocess.h"

#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

// YOLOv26PoseResult methods implementation
float YOLOv26PoseResult::iou(const YOLOv26PoseResult& other) const {
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

bool YOLOv26PoseResult::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv26PosePostProcess::YOLOv26PosePostProcess(const int input_w, const int input_h,
                                             const float score_threshold,
                                             const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    score_threshold_ = score_threshold;
    is_ort_configured_ = is_ort_configured;

    // Initialize model-specific parameters for YOLOv26Pose
    cpu_output_names_ = {"detections"};

    if (!is_ort_configured_) {
        throw std::invalid_argument(
            "ORT-OFF output postprocessing is not supported for YOLOV26Pose\n"
            "please dxrt build with USE_ORT=ON");
    }
}

// Default constructor
YOLOv26PosePostProcess::YOLOv26PosePostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    score_threshold_ = 0.5;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for YOLOv26Pose
    cpu_output_names_ = {"detections"};
}

// Process model outputs (NMS-free, score thresholding only)
std::vector<YOLOv26PoseResult> YOLOv26PosePostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    std::vector<YOLOv26PoseResult> detections;
    if (!is_ort_configured_) {
        throw std::runtime_error("YOLOv26PosePostProcess currently supports only ORT inference mode.");
    }

    if (outputs.empty()) return detections;

    // YOLOv26Pose exports final detections directly (NMS-free) in a single
    // tensor named "detections".
    //   shape: [1, N, 57]
    // where 57 = [x1, y1, x2, y2, score, class_id, 17 * 3 keypoints]

    const auto &tensor = outputs.front();
    const float *data = static_cast<const float *>(tensor->data());
    const auto &shape = tensor->shape();
    if (shape.size() < 3) return detections;

    const int num_dets = static_cast<int>(shape[1]);
    const int vec_size = static_cast<int>(shape[2]);

    if (vec_size <= 0) return detections;

    detections.reserve(num_dets);

    for (int i = 0; i < num_dets; ++i) {
        const float *det = data + i * vec_size;

        const float score = det[4];
        if (score < score_threshold_) continue;

        YOLOv26PoseResult result;
        // x1, y1, x2, y2 already given in absolute coordinates
        result.box = {det[0], det[1], det[2], det[3]};
        result.confidence = score;

        // keypoints: 17 * 3 (x, y, conf)
        const int expected_vec = 6 + num_landmarks_ * 3;
        if (vec_size >= expected_vec) {
            result.landmarks.assign(det + 6, det + 6 + num_landmarks_ * 3);
        }

        detections.emplace_back(std::move(result));
    }

    return detections;
}

// Set thresholds
void YOLOv26PosePostProcess::set_thresholds(float score_threshold) {
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
}

// Get configuration information
std::string YOLOv26PosePostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOV26Pose PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n"
        << "  Is Ort Configured: " << (is_ort_configured_ ? "Yes" : "No") << "\n";

    for (auto& cpu_output_name : cpu_output_names_) {
        oss << "  CPU output name: " << cpu_output_name << "\n";
    }

    return oss.str();
}
