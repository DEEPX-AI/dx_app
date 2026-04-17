/**
 * @file ppu_postprocessor.hpp
 * @brief Unified PPU (Post-Processing Unit) Postprocessors for v3 interface
 * 
 * Groups all PPU-based postprocessors (hardware accelerated):
 *   - YOLOv5_PPU, YOLOv7_PPU (Detection, anchor-based)
 *   - YOLOv8_PPU (Detection, anchor-free)
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
#include "yolov8_ppu_detection_postprocessor.hpp"
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
// YOLOv3Tiny PPU Detection Postprocessor (2-scale, YOLOv3 anchors)
// ============================================================================
class YOLOv3TinyPPUPostprocessor : public IPostprocessor<DetectionResult> {
public:
    YOLOv3TinyPPUPostprocessor(int input_width = 416, int input_height = 416,
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

    std::string getModelName() const override { return "YOLOv3Tiny-PPU"; }

private:
    YOLOv3TinyPPUPostProcess impl_;
};

// ============================================================================
// YOLOv8 PPU Detection Postprocessor (anchor-free)
// ============================================================================
class YOLOv8PPUPostprocessor : public IPostprocessor<DetectionResult> {
public:
    YOLOv8PPUPostprocessor(int input_width = 640, int input_height = 640,
                           float /*obj_threshold*/ = 0.25f,
                           float score_threshold = 0.4f,
                           float nms_threshold = 0.5f,
                           bool /*is_ort_configured*/ = false)
        : impl_(input_width, input_height, score_threshold, nms_threshold) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        std::vector<DetectionResult> results = convertAll(legacy_results);
        detail::scalePPUDetectionResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOv8-PPU"; }

private:
    YOLOv8PPUPostProcess impl_;
};

// ============================================================================
// YOLOX PPU Detection Postprocessor (anchor-free, grid-based decoding)
// ============================================================================
class YOLOXPPUPostprocessor : public IPostprocessor<DetectionResult> {
public:
    YOLOXPPUPostprocessor(int input_width = 640, int input_height = 640,
                          float /*obj_threshold*/ = 0.25f,
                          float score_threshold = 0.25f,
                          float nms_threshold = 0.45f,
                          bool /*is_ort_configured*/ = false)
        : impl_(input_width, input_height, score_threshold, nms_threshold) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        std::vector<DetectionResult> results = convertAll(legacy_results);
        detail::scalePPUDetectionResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOX-PPU"; }

private:
    YOLOXPPUPostProcess impl_;
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

// ============================================================================
// YOLOv10 PPU Detection Postprocessor (anchor-free, corner format)
//
// Unlike YOLOv8/v11/v12 PPU which output center (cx, cy, w, h),
// YOLOv10 PPU outputs corner (x1, y1, x2, y2) directly in the
// DeviceBoundingBox_t x, y, w, h fields. No center-to-corner conversion.
// ============================================================================
class YOLOv10PPUPostprocessor : public IPostprocessor<DetectionResult> {
public:
    YOLOv10PPUPostprocessor(int /*input_width*/ = 640, int /*input_height*/ = 640,
                            float /*obj_threshold*/ = 0.25f,
                            float score_threshold = 0.4f,
                            float nms_threshold = 0.5f,
                            bool /*is_ort_configured*/ = false)
        : score_threshold_(score_threshold), nms_threshold_(nms_threshold) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        if (outputs.empty()) return {};
        if (outputs.front()->type() != dxrt::DataType::BBOX) {
            throw std::runtime_error("[DXAPP] YOLOv10PPUPostprocessor: expected BBOX tensor");
        }

        auto num_elements = outputs[0]->shape()[1];
        auto* raw_data = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());

        std::vector<DetectionResult> results;
        for (int i = 0; i < num_elements; ++i) {
            const auto& b = raw_data[i];
            if (b.score < score_threshold_) continue;

            // Corner format: x=x1, y=y1, w=x2, h=y2
            DetectionResult det;
            det.box = {b.x, b.y, b.w, b.h};
            det.confidence = b.score;
            det.class_id = static_cast<int>(b.label);
            det.class_name = dxapp::common::get_coco_class_name(det.class_id);
            results.push_back(std::move(det));
        }

        // Apply class-agnostic NMS
        if (!results.empty()) {
            std::sort(results.begin(), results.end(),
                [](const DetectionResult& a, const DetectionResult& b) {
                    return a.confidence > b.confidence;
                });
            std::vector<bool> suppressed(results.size(), false);
            std::vector<DetectionResult> kept;
            for (size_t i = 0; i < results.size(); ++i) {
                if (suppressed[i]) continue;
                kept.push_back(results[i]);
                const auto& bi = results[i].box;
                float ai = (bi[2]-bi[0])*(bi[3]-bi[1]);
                for (size_t j = i+1; j < results.size(); ++j) {
                    if (suppressed[j]) continue;
                    const auto& bj = results[j].box;
                    float ix = std::max(0.0f, std::min(bi[2],bj[2]) - std::max(bi[0],bj[0]));
                    float iy = std::max(0.0f, std::min(bi[3],bj[3]) - std::max(bi[1],bj[1]));
                    float inter = ix * iy;
                    float aj = (bj[2]-bj[0])*(bj[3]-bj[1]);
                    float iou = inter / (ai + aj - inter + 1e-6f);
                    if (iou > nms_threshold_) suppressed[j] = true;
                }
            }
            results = std::move(kept);
        }

        detail::scalePPUDetectionResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOv10-PPU"; }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // PPU_POSTPROCESSOR_HPP
