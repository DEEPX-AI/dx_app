/**
 * @file tddfa_postprocessor.hpp
 * @brief 3DDFA v2 face alignment postprocessor
 *
 * Processes 3DMM parameter output from 3DDFA v2 models.
 * Supports multiple backbone variants (MobileNet 0.5, MobileNetV1, ResNet22).
 *
 * Output: Single tensor [1, 62] containing:
 *   - 12 pose parameters (rotation + translation)
 *   - 40 shape parameters (3DMM shape basis coefficients)
 *   - 10 expression parameters (3DMM expression basis coefficients)
 *
 * Landmark reconstruction uses BFM (Basel Face Model) 68-landmark data.
 */

#ifndef DXAPP_TDDFA_POSTPROCESSOR_HPP
#define DXAPP_TDDFA_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/bfm_68_data.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dxapp {

class TDDFAPostprocessor : public IPostprocessor<FaceAlignmentResult> {
public:
    TDDFAPostprocessor(int input_width, int input_height)
        : input_width_(input_width), input_height_(input_height) {}

    std::vector<FaceAlignmentResult> process(
        const dxrt::TensorPtrs& outputs,
        const PreprocessContext& ctx) override {
        
        std::vector<FaceAlignmentResult> results;
        if (outputs.empty()) return results;

        auto tensor = outputs[0];
        const float* raw = static_cast<const float*>(tensor->data());
        int total = static_cast<int>(tensor->size_in_bytes() / sizeof(float));
        if (total < bfm::kParamDim) return results;

        FaceAlignmentResult result;
        result.params.assign(raw, raw + total);

        // Denormalize model output: param = raw * std + mean
        float params[bfm::kParamDim];
        for (int i = 0; i < bfm::kParamDim; ++i) {
            params[i] = raw[i] * bfm::kParam_std[i] + bfm::kParam_mean[i];
        }

        // Parse 3x4 affine → R (3x3) + offset (3x1)
        float R[3][3], offset[3];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                R[r][c] = params[r * 4 + c];
            }
            offset[r] = params[r * 4 + 3];
        }

        // Extract Euler angles from R
        {
            float sy = std::max(-1.0f, std::min(1.0f, R[2][0]));
            float pitch = std::asin(sy);
            float cp = std::cos(pitch);
            float yaw, roll;
            if (std::abs(cp) > 1e-6f) {
                yaw  = std::atan2(R[2][1] / cp, R[2][2] / cp);
                roll = std::atan2(R[1][0] / cp, R[0][0] / cp);
            } else {
                yaw  = 0.0f;
                roll = std::atan2(-R[0][1], R[1][1]);
            }
            constexpr float RAD2DEG = 180.0f / 3.14159265358979f;
            result.pose = {yaw * RAD2DEG, pitch * RAD2DEG, roll * RAD2DEG};
        }

        // Reconstruct 68 landmarks using BFM
        result.landmarks_2d = reconstruct_landmarks(params, R, offset, ctx);

        results.push_back(result);
        return results;
    }

    std::string getModelName() const override { return "3DDFA-v2"; }

private:
    int input_width_;
    int input_height_;

    std::vector<Keypoint> reconstruct_landmarks(
        const float* params, const float R[3][3], const float offset[3],
        const PreprocessContext& ctx) {
        
        const float* alpha_shp = params + bfm::kPoseDim;                  // 40 values
        const float* alpha_exp = params + bfm::kPoseDim + bfm::kShapeDim; // 10 values

        // Compute deformed vertices: v = u + W_shp @ alpha_shp + W_exp @ alpha_exp
        // u_base: (3, 68) row-major, w_shp_base: (3, 68, 40), w_exp_base: (3, 68, 10)
        float verts[3][68];
        for (int d = 0; d < 3; ++d) {
            for (int j = 0; j < 68; ++j) {
                float v = bfm::kU_base[d * 68 + j];
                // Shape deformation
                for (int k = 0; k < bfm::kShapeDim; ++k) {
                    v += bfm::kW_shp_base[(d * 68 + j) * bfm::kShapeDim + k] * alpha_shp[k];
                }
                // Expression deformation
                for (int k = 0; k < bfm::kExpDim; ++k) {
                    v += bfm::kW_exp_base[(d * 68 + j) * bfm::kExpDim + k] * alpha_exp[k];
                }
                verts[d][j] = v;
            }
        }

        // Project: pts3d = R @ verts + offset
        std::vector<Keypoint> lmks(68);
        float sx = ctx.original_width  > 0 ? static_cast<float>(ctx.original_width)  / input_width_  : 1.0f;
        float sy = ctx.original_height > 0 ? static_cast<float>(ctx.original_height) / input_height_ : 1.0f;

        for (int j = 0; j < 68; ++j) {
            float px = R[0][0]*verts[0][j] + R[0][1]*verts[1][j] + R[0][2]*verts[2][j] + offset[0];
            float py = R[1][0]*verts[0][j] + R[1][1]*verts[1][j] + R[1][2]*verts[2][j] + offset[1];
            // y-flip: BFM y-up → image y-down
            px -= 1.0f;
            py = static_cast<float>(input_height_) - py;
            lmks[j] = Keypoint(px * sx, py * sy);
        }

        return lmks;
    }
};

}  // namespace dxapp

#endif  // DXAPP_TDDFA_POSTPROCESSOR_HPP
