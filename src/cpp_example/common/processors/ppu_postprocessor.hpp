/**
 * @file ppu_postprocessor.hpp
 * @brief Unified PPU (Post-Processing Unit) Postprocessors for v3 interface
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Groups all PPU-based postprocessors (hardware accelerated):
 *   - YOLOv5_PPU, YOLOv7_PPU (Detection)
 *   - YOLOv5Pose_PPU (Pose)
 *   - SCRFD_PPU (Face)
 * 
 * Note: PPU models have is_ort_configured parameter but it may not be used
 *       since PPU handles postprocessing on hardware.
 */

#ifndef PPU_POSTPROCESSOR_HPP
#define PPU_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/result_converters.hpp"

// Import shared scale functions from non-PPU group headers
#include "common/processors/yolo_detection_postprocessor.hpp"
#include "common/processors/face_postprocessor.hpp"
#include "common/processors/pose_postprocessor.hpp"

// Merged/unique postprocess headers
#include "ppu_detection_postprocessor.hpp"
#include "anchor_pose_ppu_postprocessor.hpp"
#include "scrfd_face_ppu_postprocessor.hpp"

namespace dxapp {

// ============================================================================
// PPU Scale Functions — reuse shared implementations
// (Previously duplicated from yolo_detection/face/pose_postprocessor.hpp)
// ============================================================================
namespace detail {

inline void scalePPUDetectionResults(std::vector<DetectionResult>& results,
                                      const PreprocessContext& ctx) {
    scaleDetectionResults(results, ctx);
}

inline void scalePPUPoseResults(std::vector<PoseResult>& results,
                                 const PreprocessContext& ctx) {
    scalePoseResults(results, ctx);
}

inline void scalePPUFaceResults(std::vector<FaceDetectionResult>& results,
                                 const PreprocessContext& ctx) {
    scaleFaceResults(results, ctx);
}

}  // namespace detail

// ============================================================================
// YOLOv5 PPU Detection Postprocessor
// ============================================================================
class YOLOv5PPUPostprocessor : public IPostprocessor<DetectionResult> {
public:
    YOLOv5PPUPostprocessor(int input_width = 640, int input_height = 640,
                           float obj_threshold = 0.25f,
                           float score_threshold = 0.25f,
                           float nms_threshold = 0.45f,
                           bool /*is_ort_configured*/ = false)
        : impl_(input_width, input_height, obj_threshold, score_threshold,
                nms_threshold) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        std::vector<DetectionResult> results = convertAll(legacy_results);
        detail::scalePPUDetectionResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOv5-PPU"; }

private:
    YOLOv5PPUPostProcess impl_;
};

// ============================================================================
// YOLOv7 PPU Detection Postprocessor
// ============================================================================
class YOLOv7PPUPostprocessor : public IPostprocessor<DetectionResult> {
public:
    YOLOv7PPUPostprocessor(int input_width = 640, int input_height = 640,
                           float obj_threshold = 0.25f,
                           float score_threshold = 0.25f,
                           float nms_threshold = 0.45f,
                           bool /*is_ort_configured*/ = false)
        : impl_(input_width, input_height, obj_threshold, score_threshold,
                nms_threshold) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        std::vector<DetectionResult> results = convertAll(legacy_results);
        detail::scalePPUDetectionResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOv7-PPU"; }

private:
    YOLOv7PPUPostProcess impl_;
};

// ============================================================================
// YOLOv5Pose PPU Pose Postprocessor
// ============================================================================
class YOLOv5PosePPUPostprocessor : public IPostprocessor<PoseResult> {
public:
    YOLOv5PosePPUPostprocessor(int input_width = 640, int input_height = 640,
                               float /*obj_threshold*/ = 0.5f,
                               float score_threshold = 0.5f,
                               float nms_threshold = 0.45f,
                               bool /*is_ort_configured*/ = false)
        : impl_(input_width, input_height, score_threshold,
                nms_threshold) {}

    std::vector<PoseResult> process(const dxrt::TensorPtrs& outputs,
                                    const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        auto results = convertAllWith(legacy_results,
            [](const YOLOv5PosePPUResult& s) { return convertToPose(s); });
        detail::scalePPUPoseResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOv5Pose-PPU"; }

private:
    YOLOv5PosePPUPostProcess impl_;
};

// ============================================================================
// SCRFD PPU Face Detection Postprocessor
// ============================================================================
class SCRFDPPUPostprocessor : public IPostprocessor<FaceDetectionResult> {
public:
    SCRFDPPUPostprocessor(int input_width = 640, int input_height = 640,
                          float score_threshold = 0.5f,
                          float nms_threshold = 0.45f,
                          bool /*is_ort_configured*/ = false)
        : impl_(input_width, input_height, score_threshold, nms_threshold) {}

    std::vector<FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                             const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        auto results = convertAllWith(legacy_results,
            [](const SCRFDPPUResult& s) { return convertToFace(s); });
        detail::scalePPUFaceResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "SCRFD-PPU"; }

private:
    SCRFDPPUPostProcess impl_;
};

}  // namespace dxapp

#endif  // PPU_POSTPROCESSOR_HPP
