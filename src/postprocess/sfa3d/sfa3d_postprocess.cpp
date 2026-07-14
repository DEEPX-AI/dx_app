#include "sfa3d_postprocess.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kNumClasses = 3;
constexpr float kXMin = 0.0f;
constexpr float kXMax = 50.0f;
constexpr float kYMin = -25.0f;
constexpr float kYMax = 25.0f;
constexpr float kVeloZMin = -2.73f;

inline float sigmoid(float x) {
    x = std::max(-50.0f, std::min(x, 50.0f));
    return 1.0f / (1.0f + std::exp(-x));
}

struct Peak {
    int cls{0};
    float score{0.0f};
    int row{0};
    int col{0};
};

std::vector<Peak> topkHeatmap(const float* heatmap, int num_classes, int h, int w,
                              int k, int max_k) {
    std::vector<Peak> peaks;
    peaks.reserve(static_cast<size_t>(k * num_classes));
    for (int cls = 0; cls < num_classes; ++cls) {
        std::vector<std::pair<float, int>> scored;
        scored.reserve(static_cast<size_t>(h * w));
        const float* cls_map = heatmap + cls * h * w;
        for (int idx = 0; idx < h * w; ++idx) {
            scored.emplace_back(cls_map[idx], idx);
        }
        const int take = std::min(max_k, static_cast<int>(scored.size()));
        std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < take; ++i) {
            const int flat = scored[static_cast<size_t>(i)].second;
            peaks.push_back({cls, scored[static_cast<size_t>(i)].first,
                             flat / w, flat % w});
        }
    }
    return peaks;
}

const float* tensorData(const dxrt::TensorPtr& tensor) {
    return static_cast<const float*>(tensor->data());
}

int tensorChannels(const dxrt::TensorPtr& tensor) {
    const auto shape = tensor->shape();
    if (shape.size() < 3) return -1;
    return static_cast<int>(shape[shape.size() - 3]);
}

bool isSfa3dFiveHeadLayout(const dxrt::TensorPtrs& outputs) {
    if (outputs.size() < 5) return false;
    return tensorChannels(outputs[0]) == kNumClasses &&
           tensorChannels(outputs[1]) == 2 &&
           tensorChannels(outputs[2]) == 2 &&
           tensorChannels(outputs[3]) == 1 &&
           tensorChannels(outputs[4]) == kNumClasses;
}

}  // namespace

SFA3DPostProcess::SFA3DPostProcess(int input_w, int input_h,
                                   float score_threshold, float nms_threshold)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold) {}

SFA3DPostProcess::SFA3DPostProcess()
    : input_width_(608), input_height_(608),
      score_threshold_(0.3f), nms_threshold_(0.2f) {}

std::vector<SFA3DResult> SFA3DPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) return {};

    // Multi-head layout: hm(3), offset(2), direction(2), z(1), dim(3)
    if (outputs.size() >= 5) {
        const dxrt::TensorPtr* hm_tensor = nullptr;
        const dxrt::TensorPtr* offset_tensor = nullptr;
        const dxrt::TensorPtr* direction_tensor = nullptr;
        const dxrt::TensorPtr* z_tensor = nullptr;
        const dxrt::TensorPtr* dim_tensor = nullptr;

        if (isSfa3dFiveHeadLayout(outputs)) {
            hm_tensor = &outputs[0];
            offset_tensor = &outputs[1];
            direction_tensor = &outputs[2];
            z_tensor = &outputs[3];
            dim_tensor = &outputs[4];
        } else {
            std::unordered_map<int, std::vector<const dxrt::TensorPtr*>> by_channels;
            for (size_t i = 0; i < outputs.size() && i < 5; ++i) {
                const int ch = tensorChannels(outputs[i]);
                if (ch > 0) {
                    by_channels[ch].push_back(&outputs[i]);
                }
            }
            auto hm_it = by_channels.find(kNumClasses);
            auto two_it = by_channels.find(2);
            auto one_it = by_channels.find(1);
            if (hm_it != by_channels.end() && hm_it->second.size() >= 2 &&
                two_it != by_channels.end() && two_it->second.size() >= 2 &&
                one_it != by_channels.end() && !one_it->second.empty()) {
                hm_tensor = hm_it->second[0];
                dim_tensor = hm_it->second[1];
                offset_tensor = two_it->second[0];
                direction_tensor = two_it->second[1];
                z_tensor = one_it->second[0];
            }
        }

        if (hm_tensor != nullptr && offset_tensor != nullptr &&
            direction_tensor != nullptr && z_tensor != nullptr &&
            dim_tensor != nullptr) {
            const auto& hm_shape = (*hm_tensor)->shape();
            const int h = static_cast<int>(hm_shape[hm_shape.size() - 2]);
            const int w = static_cast<int>(hm_shape[hm_shape.size() - 1]);
            const int hw = h * w;

            std::vector<float> heatmap(static_cast<size_t>(kNumClasses * hw));
            for (int cls = 0; cls < kNumClasses; ++cls) {
                const float* src = tensorData(*hm_tensor) + cls * hw;
                for (int j = 0; j < hw; ++j) {
                    heatmap[static_cast<size_t>(cls * hw + j)] = sigmoid(src[j]);
                }
            }

            const float* offset = tensorData(*offset_tensor);
            const float* direction = tensorData(*direction_tensor);
            const float* z_coord = tensorData(*z_tensor);
            const float* dims = tensorData(*dim_tensor);

            const float x_res = (kXMax - kXMin) / static_cast<float>(h);
            const float y_res = (kYMax - kYMin) / static_cast<float>(w);
            const float scale_x = static_cast<float>(input_width_) / std::max(w, 1);
            const float scale_y = static_cast<float>(input_height_) / std::max(h, 1);

            auto peaks = topkHeatmap(heatmap.data(), kNumClasses, h, w,
                                     max_detections_, max_detections_);
            std::vector<SFA3DResult> results;
            for (const auto& peak : peaks) {
                if (peak.score < score_threshold_) continue;
                const int row = peak.row;
                const int col = peak.col;
                const int idx = row * w + col;

                SFA3DResult det;
                det.class_id = peak.cls;
                det.confidence = peak.score;
                det.bev_x = (col + offset[idx]) * scale_x;
                det.bev_y = (row + offset[hw + idx]) * scale_y;
                det.x3d = kXMax - (row + offset[hw + idx]) * x_res;
                det.y3d = kYMin + (col + offset[idx]) * y_res;
                det.z3d = z_coord[idx] + kVeloZMin;
                det.dim_h = dims[idx];
                det.dim_w = dims[hw + idx];
                det.dim_l = dims[2 * hw + idx];
                det.yaw = std::atan2(direction[idx], direction[hw + idx]);
                det.bev_w = (det.dim_w / y_res) * scale_x;
                det.bev_h = (det.dim_l / x_res) * scale_y;
                results.push_back(det);
            }
            std::sort(results.begin(), results.end(),
                      [](const SFA3DResult& a, const SFA3DResult& b) {
                          return a.confidence > b.confidence;
                      });
            if (static_cast<int>(results.size()) > max_detections_) {
                results.resize(static_cast<size_t>(max_detections_));
            }
            return results;
        }
    }

    // Fused single tensor fallback
    const auto shape = outputs[0]->shape();
    if (shape.size() < 3) return {};

    int c = static_cast<int>(shape[shape.size() - 3]);
    int h = static_cast<int>(shape[shape.size() - 2]);
    int w = static_cast<int>(shape[shape.size() - 1]);
    const float* data = tensorData(outputs[0]);
    const int hw = h * w;

    if (c < kNumClasses + 8) {
        std::vector<float> heatmap(static_cast<size_t>(hw));
        for (int j = 0; j < hw; ++j) {
            heatmap[static_cast<size_t>(j)] = sigmoid(data[j]);
        }
        auto peaks = topkHeatmap(heatmap.data(), 1, h, w, max_detections_, max_detections_);
        std::vector<SFA3DResult> results;
        const float x_res = (kXMax - kXMin) / static_cast<float>(h);
        const float y_res = (kYMax - kYMin) / static_cast<float>(w);
        const float scale_x = static_cast<float>(input_width_) / std::max(w, 1);
        const float scale_y = static_cast<float>(input_height_) / std::max(h, 1);
        for (const auto& peak : peaks) {
            if (peak.score < score_threshold_) continue;
            SFA3DResult det;
            det.class_id = 0;
            det.confidence = peak.score;
            det.bev_x = static_cast<float>(peak.col) * scale_x;
            det.bev_y = static_cast<float>(peak.row) * scale_y;
            det.x3d = kXMax - peak.row * x_res;
            det.y3d = kYMin + peak.col * y_res;
            det.bev_w = 10.0f;
            det.bev_h = 10.0f;
            results.push_back(det);
        }
        return results;
    }

    std::vector<float> heatmap(static_cast<size_t>(kNumClasses * hw));
    for (int cls = 0; cls < kNumClasses; ++cls) {
        for (int j = 0; j < hw; ++j) {
            heatmap[static_cast<size_t>(cls * hw + j)] =
                sigmoid(data[cls * hw + j]);
        }
    }
    const float* offset = data + kNumClasses * hw;
    const float* z_coord = data + (kNumClasses + 2) * hw;
    const float* dims = data + (kNumClasses + 3) * hw;
    const float* yaw_im = data + (kNumClasses + 6) * hw;
    const float* yaw_re = data + (kNumClasses + 7) * hw;

    const float x_res = (kXMax - kXMin) / static_cast<float>(h);
    const float y_res = (kYMax - kYMin) / static_cast<float>(w);
    const float scale_x = static_cast<float>(input_width_) / std::max(w, 1);
    const float scale_y = static_cast<float>(input_height_) / std::max(h, 1);

    auto peaks = topkHeatmap(heatmap.data(), kNumClasses, h, w,
                             max_detections_, max_detections_);
    std::vector<SFA3DResult> results;
    for (const auto& peak : peaks) {
        if (peak.score < score_threshold_) continue;
        const int idx = peak.row * w + peak.col;

        SFA3DResult det;
        det.class_id = peak.cls;
        det.confidence = peak.score;
        det.bev_x = (peak.col + offset[idx]) * scale_x;
        det.bev_y = (peak.row + offset[hw + idx]) * scale_y;
        det.x3d = kXMax - (peak.row + offset[hw + idx]) * x_res;
        det.y3d = kYMin + (peak.col + offset[idx]) * y_res;
        det.z3d = z_coord[idx] + kVeloZMin;
        det.dim_h = dims[idx];
        det.dim_w = dims[hw + idx];
        det.dim_l = dims[2 * hw + idx];
        det.yaw = std::atan2(yaw_im[idx], yaw_re[idx]);
        det.bev_w = (det.dim_w / y_res) * scale_x;
        det.bev_h = (det.dim_l / x_res) * scale_y;
        results.push_back(det);
    }
    std::sort(results.begin(), results.end(),
              [](const SFA3DResult& a, const SFA3DResult& b) {
                  return a.confidence > b.confidence;
              });
    if (static_cast<int>(results.size()) > max_detections_) {
        results.resize(static_cast<size_t>(max_detections_));
    }
    return results;
}
