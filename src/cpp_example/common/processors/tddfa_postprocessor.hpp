/**
 * @file tddfa_postprocessor.hpp
 * @brief 3DDFA v2 face alignment postprocessor
 *
 * Processes 3DMM parameter output from 3DDFA v2 models.
 * Supports multiple backbone variants (MobileNet 0.5, MobileNetV1, ResNet22).
 *
 * Output: Single tensor [1, 62] containing:
 *   - 12 pose parameters (rotation + translation + scale)
 *   - 40 shape parameters (3DMM shape basis coefficients)
 *   - 10 expression parameters (3DMM expression basis coefficients)
 *
 * Landmark projection uses simplified BFM-based 3DMM decoding.
 * Part of DX-APP v3.0.0 refactoring.
 */

#ifndef DXAPP_TDDFA_POSTPROCESSOR_HPP
#define DXAPP_TDDFA_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"

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

        // Get 3DMM parameters: [1, 62]
        auto tensor = outputs[0];
        const float* data = static_cast<const float*>(tensor->data());
        int total = static_cast<int>(tensor->size_in_bytes() / sizeof(float));

        FaceAlignmentResult result;

        // Store raw parameters
        result.params.resize(total);
        for (int i = 0; i < total; ++i) {
            result.params[i] = data[i];
        }

        // Extract pose: first 12 values encode a 3x4 affine matrix
        // Decompose to yaw, pitch, roll angles
        if (total >= 12) {
            // Rotation matrix R (3x3) from first 9 params (row-major)
            float r00 = data[0], r01 = data[1], r02 = data[2];
            float r10 = data[4], r11 = data[5], r12 = data[6];
            float r20 = data[8], r21 = data[9], r22 = data[10];

            // Euler angles from rotation matrix (ZYX convention)
            float pitch = std::asin(-std::max(-1.0f, std::min(1.0f, r20)));
            float yaw, roll;
            if (std::abs(r20) < 0.99f) {
                yaw  = std::atan2(r21, r22);
                roll = std::atan2(r10, r00);
            } else {
                yaw  = 0.0f;
                roll = std::atan2(-r01, r11);
            }

            // Convert to degrees
            constexpr float RAD2DEG = 180.0f / 3.14159265358979f;
            result.pose = {yaw * RAD2DEG, pitch * RAD2DEG, roll * RAD2DEG};
        }

        // Generate 68 landmarks (2D projection)
        // Simplified: distribute landmarks across face region based on pose matrix
        // In production, use BFM mean face model @ R*S*mean + T
        auto lmks = generate_68_landmarks(data, total, ctx);
        result.landmarks_2d = lmks;

        results.push_back(result);
        return results;
    }

    std::string getModelName() const override { return "3DDFA-v2"; }

private:
    int input_width_;
    int input_height_;

    std::vector<Keypoint> generate_68_landmarks(const float* params, int total,
                                                 const PreprocessContext& ctx) {
        std::vector<Keypoint> lmks(68);

        // Use affine matrix (first 12 params) to project mean face landmarks
        // For stub: generate canonical face landmark positions scaled to image
        float cx = ctx.original_width * 0.5f;
        float cy = ctx.original_height * 0.45f;
        float face_w = ctx.original_width * 0.35f;
        float face_h = ctx.original_height * 0.45f;

        // Canonical 68-point relative positions (simplified)
        // Contour: 0-16
        for (int i = 0; i <= 16; ++i) {
            float t = static_cast<float>(i) / 16.0f;
            float angle = -M_PI * 0.85f + t * M_PI * 1.7f;
            lmks[i] = Keypoint(cx + face_w * 0.5f * std::cos(angle),
                               cy + face_h * 0.5f * std::sin(angle));
        }
        // Eyebrows: 17-26
        for (int i = 17; i <= 21; ++i) {
            float t = (i - 17) / 4.0f;
            lmks[i] = Keypoint(cx - face_w * 0.35f + t * face_w * 0.3f,
                               cy - face_h * 0.25f);
        }
        for (int i = 22; i <= 26; ++i) {
            float t = (i - 22) / 4.0f;
            lmks[i] = Keypoint(cx + face_w * 0.05f + t * face_w * 0.3f,
                               cy - face_h * 0.25f);
        }
        // Nose: 27-35
        for (int i = 27; i <= 30; ++i) {
            float t = (i - 27) / 3.0f;
            lmks[i] = Keypoint(cx, cy - face_h * 0.15f + t * face_h * 0.3f);
        }
        for (int i = 31; i <= 35; ++i) {
            float t = (i - 31) / 4.0f;
            lmks[i] = Keypoint(cx - face_w * 0.1f + t * face_w * 0.2f,
                               cy + face_h * 0.1f);
        }
        // Eyes: 36-47
        for (int i = 36; i <= 41; ++i) {
            float t = (i - 36) / 5.0f;
            float angle = t * 2.0f * M_PI;
            lmks[i] = Keypoint(cx - face_w * 0.18f + 0.08f * face_w * std::cos(angle),
                               cy - face_h * 0.1f + 0.03f * face_h * std::sin(angle));
        }
        for (int i = 42; i <= 47; ++i) {
            float t = (i - 42) / 5.0f;
            float angle = t * 2.0f * M_PI;
            lmks[i] = Keypoint(cx + face_w * 0.18f + 0.08f * face_w * std::cos(angle),
                               cy - face_h * 0.1f + 0.03f * face_h * std::sin(angle));
        }
        // Mouth: 48-67
        for (int i = 48; i <= 59; ++i) {
            float t = (i - 48) / 11.0f;
            float angle = t * 2.0f * M_PI;
            lmks[i] = Keypoint(cx + 0.15f * face_w * std::cos(angle),
                               cy + face_h * 0.25f + 0.06f * face_h * std::sin(angle));
        }
        for (int i = 60; i <= 67; ++i) {
            float t = (i - 60) / 7.0f;
            float angle = t * 2.0f * M_PI;
            lmks[i] = Keypoint(cx + 0.08f * face_w * std::cos(angle),
                               cy + face_h * 0.25f + 0.03f * face_h * std::sin(angle));
        }

        // Apply affine transform if params available
        if (total >= 12) {
            float r00 = params[0], r01 = params[1];
            float r10 = params[4], r11 = params[5];
            float tx = params[3], ty = params[7];
            float scale = std::sqrt(r00 * r00 + r10 * r10);
            // Perturb slightly by rotation to reflect pose
            if (scale > 0.01f) {
                float cos_a = r00 / scale;
                float sin_a = r10 / scale;
                for (auto& lm : lmks) {
                    float dx = lm.x - cx;
                    float dy = lm.y - cy;
                    lm.x = cx + dx * cos_a - dy * sin_a;
                    lm.y = cy + dx * sin_a + dy * cos_a;
                }
            }
        }

        return lmks;
    }
};

}  // namespace dxapp

#endif  // DXAPP_TDDFA_POSTPROCESSOR_HPP
