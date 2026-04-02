/**
 * @file result_converters.hpp
 * @brief Conversion functions between legacy postprocess results and common result types
 * 
 * Part of DX-APP v3.0.0 refactoring. These functions convert model-specific
 * result structures to common result types for unified processing.
 */

#ifndef RESULT_CONVERTERS_HPP
#define RESULT_CONVERTERS_HPP

#include "common/base/i_processor.hpp"

// Merged postprocess headers
#include "anchor_detection_postprocessor.hpp"
#include "anchorless_dfl_detection_postprocessor.hpp"
#include "ppu_detection_postprocessor.hpp"
// Unique postprocess headers
#include "scrfd_face_postprocessor.hpp"
#include "anchor_face_postprocessor.hpp"
#include "anchor_pose_postprocessor.hpp"
#include "anchorless_instance_seg_postprocessor.hpp"
#include "argmax_semantic_seg_postprocessor.hpp"
#include "anchor_pose_ppu_postprocessor.hpp"
#include "scrfd_face_ppu_postprocessor.hpp"

namespace dxapp {

// ============================================================================
// Common Coordinate Scaling Utilities
//
// Generic helpers for the recurring pattern:
//   coord = (coord - pad) / scale;  coord = clamp(coord, 0, original_dim);
//
// Used by scaleDetectionResults, scaleFaceResults, scalePoseResults,
// scaleOBBResults, scaleInstanceSegResults.
// ============================================================================
namespace detail {

/**
 * @brief Scale a [x1, y1, x2, y2] bounding box from preprocessed to original coords.
 */
inline void scaleBox(std::vector<float>& box, const PreprocessContext& ctx) {
    if (box.size() < 4) return;
    box[0] = (box[0] - ctx.pad_x) / ctx.scale;
    box[1] = (box[1] - ctx.pad_y) / ctx.scale;
    box[2] = (box[2] - ctx.pad_x) / ctx.scale;
    box[3] = (box[3] - ctx.pad_y) / ctx.scale;

    float w = static_cast<float>(ctx.original_width);
    float h = static_cast<float>(ctx.original_height);
    box[0] = std::max(0.0f, std::min(box[0], w));
    box[1] = std::max(0.0f, std::min(box[1], h));
    box[2] = std::max(0.0f, std::min(box[2], w));
    box[3] = std::max(0.0f, std::min(box[3], h));
}

/**
 * @brief Scale a single keypoint from preprocessed to original coords.
 */
inline void scaleKeypoint(Keypoint& kp, const PreprocessContext& ctx) {
    kp.x = (kp.x - ctx.pad_x) / ctx.scale;
    kp.y = (kp.y - ctx.pad_y) / ctx.scale;
}

}  // namespace detail

// ============================================================================
// Detection Result Converters
// AnchorYOLOResult and AnchorlessYOLOResult both have: box, confidence, class_id, class_name
// ============================================================================

inline DetectionResult convert(const AnchorYOLOResult& src) {
    return DetectionResult(src.box, src.confidence, src.class_id, src.class_name);
}

inline DetectionResult convert(const AnchorlessYOLOResult& src) {
    return DetectionResult(src.box, src.confidence, src.class_id, src.class_name);
}

// ============================================================================
// PPU Detection Result Converters
// ============================================================================

inline DetectionResult convert(const PPUDetectionResult& src) {
    return DetectionResult(src.box, src.confidence, src.class_id, src.class_name);
}

// ============================================================================
// Face Detection Result Converters
// Legacy structs use: box(vector<float>), confidence, landmarks(vector<float>)
// landmarks is flat: [x0,y0,x1,y1,...,x4,y4] (5 points, stride 2)
// ============================================================================

inline FaceDetectionResult convertToFace(const SCRFDResult& src) {
    FaceDetectionResult result;
    result.box = src.box;
    result.confidence = src.confidence;
    
    // Convert flat landmarks to Keypoint vector (5 points, stride 2)
    for (size_t i = 0; i + 1 < src.landmarks.size(); i += 2) {
        result.landmarks.emplace_back(src.landmarks[i], src.landmarks[i + 1]);
    }
    
    return result;
}

inline FaceDetectionResult convertToFace(const YOLOv5FaceResult& src) {
    FaceDetectionResult result;
    result.box = src.box;
    result.confidence = src.confidence;
    
    // Convert flat landmarks to Keypoint vector
    for (size_t i = 0; i + 1 < src.landmarks.size(); i += 2) {
        result.landmarks.emplace_back(src.landmarks[i], src.landmarks[i + 1]);
    }
    
    return result;
}

// PPU Face: SCRFDPPUResult has same structure as SCRFDResult
inline FaceDetectionResult convertToFace(const SCRFDPPUResult& src) {
    FaceDetectionResult result;
    result.box = src.box;
    result.confidence = src.confidence;
    
    for (size_t i = 0; i + 1 < src.landmarks.size(); i += 2) {
        result.landmarks.emplace_back(src.landmarks[i], src.landmarks[i + 1]);
    }
    
    return result;
}

// ============================================================================
// Pose Estimation Result Converters
// Legacy struct uses: box(vector<float>), confidence, landmarks(vector<float>)
// landmarks is flat: [x0,y0,c0,x1,y1,c1,...] (17 points, stride 3)
// ============================================================================

inline PoseResult convertToPose(const YOLOv5PoseResult& src) {
    PoseResult result;
    result.box = src.box;
    result.confidence = src.confidence;
    
    // Convert flat landmarks to Keypoint vector (17 points, stride 3 with confidence)
    for (size_t i = 0; i + 2 < src.landmarks.size(); i += 3) {
        result.keypoints.emplace_back(src.landmarks[i], src.landmarks[i + 1], src.landmarks[i + 2]);
    }
    
    return result;
}

// PPU Pose: YOLOv5PosePPUResult has same structure as YOLOv5PoseResult
inline PoseResult convertToPose(const YOLOv5PosePPUResult& src) {
    PoseResult result;
    result.box = src.box;
    result.confidence = src.confidence;
    
    // PPU landmarks: 17 points, stride 3 (x, y, conf)
    for (size_t i = 0; i + 2 < src.landmarks.size(); i += 3) {
        result.keypoints.emplace_back(src.landmarks[i], src.landmarks[i + 1], src.landmarks[i + 2]);
    }
    
    return result;
}

// ============================================================================
// Instance Segmentation Result Converters
// Legacy struct uses: box, confidence, class_id, class_name,
//   mask(vector<float>), mask_height, mask_width
// ============================================================================

inline InstanceSegmentationResult convertToInstanceSeg(const YOLOv8SegResult& src) {
    InstanceSegmentationResult result;
    result.box = src.box;
    result.confidence = src.confidence;
    result.class_id = src.class_id;
    result.class_name = src.class_name;
    
    // Convert flat mask vector to cv::Mat
    if (!src.mask.empty() && src.mask_height > 0 && src.mask_width > 0) {
        std::vector<float> mask_data(src.mask);  // copy to obtain non-const ptr for cv::Mat ctor
        cv::Mat mask_float(src.mask_height, src.mask_width, CV_32FC1, mask_data.data());
        mask_float.convertTo(result.mask, CV_8UC1, 255.0);
    }
    
    return result;
}

// ============================================================================
// Semantic Segmentation Result Converters
// Legacy struct uses: segmentation_mask(vector<int>), class_ids, class_names,
//   width, height, num_classes
// ============================================================================

inline SegmentationResult convertToSemSeg(const DeepLabv3Result& src) {
    SegmentationResult result;
    result.mask = src.segmentation_mask;
    result.width = src.width;
    result.height = src.height;
    result.class_ids = src.class_ids;
    result.class_names = src.class_names;
    
    return result;
}

// ============================================================================
// Batch Conversion Helpers
// ============================================================================

template <typename SrcType>
std::vector<DetectionResult> convertAll(const std::vector<SrcType>& src_results) {
    std::vector<DetectionResult> results;
    results.reserve(src_results.size());
    for (const auto& src : src_results) {
        results.push_back(convert(src));
    }
    return results;
}

template <typename SrcType, typename ConvertFunc>
auto convertAllWith(const std::vector<SrcType>& src_results, ConvertFunc convert_fn) 
    -> std::vector<decltype(convert_fn(std::declval<SrcType>()))> {
    std::vector<decltype(convert_fn(std::declval<SrcType>()))> results;
    results.reserve(src_results.size());
    for (const auto& src : src_results) {
        results.push_back(convert_fn(src));
    }
    return results;
}

}  // namespace dxapp

#endif  // RESULT_CONVERTERS_HPP
