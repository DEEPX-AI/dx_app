/**
 * @file sfa3d_postprocessor.hpp
 * @brief SFA3D 3D detection postprocessor (v3 header-only wrapper)
 */

#ifndef SFA3D_POSTPROCESSOR_V3_HPP
#define SFA3D_POSTPROCESSOR_V3_HPP

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "common/base/i_processor.hpp"
#include "sfa3d_postprocess.h"

namespace dxapp {

static const std::vector<std::string> SFA3D_CLASS_NAMES = {
    "Pedestrian", "Car", "Cyclist"};

inline Detection3DResult toDetection3DResult(const SFA3DResult& src) {
    Detection3DResult det;
    det.class_id = src.class_id;
    det.confidence = src.confidence;
    det.bev_x = src.bev_x;
    det.bev_y = src.bev_y;
    det.bev_w = src.bev_w;
    det.bev_h = src.bev_h;
    det.x3d = src.x3d;
    det.y3d = src.y3d;
    det.z3d = src.z3d;
    det.dim_h = src.dim_h;
    det.dim_w = src.dim_w;
    det.dim_l = src.dim_l;
    det.yaw = src.yaw;
    if (src.class_id >= 0 &&
        src.class_id < static_cast<int>(SFA3D_CLASS_NAMES.size())) {
        det.class_name = SFA3D_CLASS_NAMES[static_cast<size_t>(src.class_id)];
    } else {
        det.class_name = "cls_" + std::to_string(src.class_id);
    }
    return det;
}

class SFA3DPostprocessor : public IPostprocessor<Detection3DResult> {
public:
    SFA3DPostprocessor(int input_width, int input_height,
                       float score_threshold = 0.3f,
                       float nms_threshold = 0.2f,
                       bool /*is_ort_configured*/ = false)
        : backend_(input_width, input_height, score_threshold, nms_threshold) {}

    std::vector<Detection3DResult> process(const dxrt::TensorPtrs& outputs,
                                           const PreprocessContext& /*ctx*/) override {
        const auto raw = backend_.postprocess(outputs);
        std::vector<Detection3DResult> results;
        results.reserve(raw.size());
        for (const auto& item : raw) {
            results.push_back(toDetection3DResult(item));
        }
        return results;
    }

    std::string getModelName() const override { return "sfa3d"; }

private:
    SFA3DPostProcess backend_;
};

}  // namespace dxapp

#endif  // SFA3D_POSTPROCESSOR_V3_HPP
