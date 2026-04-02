/**
 * @file face_postprocessor.hpp
 * @brief Unified Face Detection Postprocessors for v3 interface
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Groups all face detection postprocessors:
 *   - SCRFD (5 args): score_threshold, nms_threshold
 *   - YOLOv5Face (6 args): obj_threshold, score_threshold, nms_threshold
 */

#ifndef FACE_POSTPROCESSOR_HPP
#define FACE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/result_converters.hpp"

// Postprocess headers
#include "scrfd_face_postprocessor.hpp"
#include "anchor_face_postprocessor.hpp"

namespace dxapp {

namespace detail {

inline void scaleFaceResults(std::vector<FaceDetectionResult>& results,
                             const PreprocessContext& ctx) {
    for (auto& face : results) {
        scaleBox(face.box, ctx);
        for (auto& kp : face.landmarks) {
            scaleKeypoint(kp, ctx);
        }
    }
}

}  // namespace detail

// ============================================================================
// SCRFD Postprocessor (5 args: no obj_threshold)
// ============================================================================
class SCRFDPostprocessor : public IPostprocessor<FaceDetectionResult> {
public:
    SCRFDPostprocessor(int input_width = 640, int input_height = 640,
                       float score_threshold = 0.5f,
                       float nms_threshold = 0.4f,
                       bool is_ort_configured = false)
        : impl_(input_width, input_height, score_threshold, nms_threshold,
                is_ort_configured) {}

    std::vector<FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                             const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        auto results = convertAllWith(legacy_results,
            [](const SCRFDResult& s) { return convertToFace(s); });
        detail::scaleFaceResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "SCRFD"; }

private:
    SCRFDPostProcess impl_;
};

// ============================================================================
// YOLOv5Face Postprocessor (6 args: obj_threshold included)
// ============================================================================
class YOLOv5FacePostprocessor : public IPostprocessor<FaceDetectionResult> {
public:
    YOLOv5FacePostprocessor(int input_width = 640, int input_height = 640,
                            float obj_threshold = 0.25f,
                            float score_threshold = 0.3f,
                            float nms_threshold = 0.45f,
                            bool is_ort_configured = false)
        : impl_(input_width, input_height, obj_threshold, score_threshold,
                nms_threshold, is_ort_configured) {}

    std::vector<FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                             const PreprocessContext& ctx) override {
        auto legacy_results = impl_.postprocess(outputs);
        auto results = convertAllWith(legacy_results,
            [](const YOLOv5FaceResult& s) { return convertToFace(s); });
        detail::scaleFaceResults(results, ctx);
        return results;
    }

    std::string getModelName() const override { return "YOLOv5Face"; }

private:
    YOLOv5FacePostProcess impl_;
};

}  // namespace dxapp

#endif  // FACE_POSTPROCESSOR_HPP
