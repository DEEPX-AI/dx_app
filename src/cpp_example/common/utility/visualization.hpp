/**
 * @file visualization.hpp
 * @brief Common visualization utilities for drawing results
 */

#ifndef DXAPP_VISUALIZATION_HPP
#define DXAPP_VISUALIZATION_HPP

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <string>
#include <vector>

#include "../base/i_processor.hpp"

namespace dxapp {

/**
 * @brief Pre-computed color table for class visualization (matching original)
 */
inline const std::vector<cv::Scalar>& getCocoClassColors() {
    static const std::vector<cv::Scalar> COCO_CLASS_COLORS = {
        cv::Scalar(255, 0, 0),      // Red
        cv::Scalar(0, 255, 0),      // Green
        cv::Scalar(0, 0, 255),      // Blue
        cv::Scalar(255, 255, 0),    // Cyan
        cv::Scalar(255, 0, 255),    // Magenta
        cv::Scalar(0, 255, 255),    // Yellow
        cv::Scalar(128, 0, 128),    // Purple
        cv::Scalar(255, 165, 0),    // Orange
        cv::Scalar(0, 128, 0),      // Dark Green
        cv::Scalar(128, 128, 0),    // Olive
        cv::Scalar(0, 128, 128),    // Teal
        cv::Scalar(128, 0, 0),      // Maroon
        cv::Scalar(192, 192, 192),  // Silver
        cv::Scalar(255, 192, 203),  // Pink
        cv::Scalar(255, 215, 0),    // Gold
        cv::Scalar(173, 216, 230),  // Light Blue
        cv::Scalar(144, 238, 144),  // Light Green
        cv::Scalar(255, 218, 185),  // Peach
        cv::Scalar(221, 160, 221),  // Plum
        cv::Scalar(255, 240, 245)   // Lavender Blush
    };
    return COCO_CLASS_COLORS;
}

/**
 * @brief Generate a consistent color for a class ID (matching original)
 * @param class_id Class identifier
 * @return BGR color as cv::Scalar
 */
inline cv::Scalar getClassColor(int class_id) {
    const auto& colors = getCocoClassColors();
    return colors[class_id % colors.size()];
}

/**
 * @brief Predefined color palette for segmentation
 */
inline const std::vector<cv::Scalar>& getSegmentationPalette() {
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(128, 64, 128),   // road
        cv::Scalar(244, 35, 232),   // sidewalk
        cv::Scalar(70, 70, 70),     // building
        cv::Scalar(102, 102, 156),  // wall
        cv::Scalar(190, 153, 153),  // fence
        cv::Scalar(153, 153, 153),  // pole
        cv::Scalar(250, 170, 30),   // traffic light
        cv::Scalar(220, 220, 0),    // traffic sign
        cv::Scalar(107, 142, 35),   // vegetation
        cv::Scalar(152, 251, 152),  // terrain
        cv::Scalar(70, 130, 180),   // sky
        cv::Scalar(220, 20, 60),    // person
        cv::Scalar(255, 0, 0),      // rider
        cv::Scalar(0, 0, 142),      // car
        cv::Scalar(0, 0, 70),       // truck
        cv::Scalar(0, 60, 100),     // bus
        cv::Scalar(0, 80, 100),     // train
        cv::Scalar(0, 0, 230),      // motorcycle
        cv::Scalar(119, 11, 32)     // bicycle
    };
    return palette;
}

/**
 * @brief Draw a single detection on the image
 * @param frame Image to draw on (modified in place)
 * @param detection Detection result
 * @param ctx Preprocessing context for coordinate transformation
 * @param line_thickness Bounding box line thickness
 * @param font_scale Label font scale
 */
inline void drawDetection(cv::Mat& frame, const DetectionResult& detection,
                          const PreprocessContext& ctx,
                          int line_thickness = 2,
                          double font_scale = 0.5) {
    (void)ctx;  // Coordinates already scaled in postprocessor
    
    // Use pre-scaled coordinates directly (already in original image space)
    float x1 = detection.box[0];
    float y1 = detection.box[1];
    float x2 = detection.box[2];
    float y2 = detection.box[3];

    // Clamp to image bounds
    x1 = std::max(0.0f, std::min(x1, static_cast<float>(frame.cols)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(frame.rows)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(frame.cols)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(frame.rows)));

    cv::Scalar color = getClassColor(detection.class_id);

    // Draw bounding box
    cv::rectangle(frame, cv::Point2f(x1, y1), cv::Point2f(x2, y2), color, line_thickness);

    // Prepare label text
    std::string label = detection.class_name + ": " + 
                        std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

    // Get text size (matching original: thickness=2)
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                         font_scale, 2, &baseline);

    // Draw label background (matching original: class color, not black)
    cv::Point label_pt(static_cast<int>(x1),
                       y1 - 10 > 10 ? static_cast<int>(y1 - 10)
                                    : static_cast<int>(y1 + text_size.height + 10));

    cv::rectangle(frame,
                  cv::Point(label_pt.x, label_pt.y - text_size.height - 5),
                  cv::Point(label_pt.x + text_size.width, label_pt.y + baseline),
                  color, cv::FILLED);

    // Draw label text (matching original: thickness=2)
    cv::putText(frame, label, label_pt,
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), 2);
}

/**
 * @brief Draw multiple detections on the image
 * @param frame Image to draw on
 * @param detections Vector of detection results
 * @param ctx Preprocessing context
 * @param line_thickness Bounding box line thickness
 * @param font_scale Label font scale
 * @return Image with drawn detections
 */
inline cv::Mat drawDetections(const cv::Mat& frame,
                              const std::vector<DetectionResult>& detections,
                              const PreprocessContext& ctx,
                              int line_thickness = 2,
                              double font_scale = 0.5) {
    cv::Mat result = frame.clone();
    
    for (const auto& detection : detections) {
        drawDetection(result, detection, ctx, line_thickness, font_scale);
    }
    
    return result;
}

/**
 * @brief Draw segmentation overlay on the image
 * @param frame Original image
 * @param segmentation Segmentation result
 * @param ctx Preprocessing context
 * @param alpha Overlay transparency (0.0 to 1.0)
 * @return Image with segmentation overlay
 */
inline cv::Mat drawSegmentation(const cv::Mat& frame,
                                const SegmentationResult& segmentation,
                                const PreprocessContext& ctx,
                                float alpha = 0.6f) {
    if (segmentation.mask.empty() || segmentation.width == 0 || segmentation.height == 0) {
        return frame.clone();
    }

    const auto& palette = getSegmentationPalette();

    // Create colored mask
    cv::Mat mask_image = cv::Mat::zeros(segmentation.height, segmentation.width, CV_8UC3);
    
    for (int y = 0; y < segmentation.height; ++y) {
        for (int x = 0; x < segmentation.width; ++x) {
            int idx = y * segmentation.width + x;
            if (idx < static_cast<int>(segmentation.mask.size())) {
                int class_id = segmentation.mask[idx];
                cv::Scalar color = (class_id >= 0 && class_id < static_cast<int>(palette.size()))
                    ? palette[class_id] : cv::Scalar(0, 0, 0);
                mask_image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uchar>(color[0]),
                    static_cast<uchar>(color[1]),
                    static_cast<uchar>(color[2])
                );
            }
        }
    }

    // Remove letterbox padding and resize to original size
    int unpad_w = segmentation.width - 2 * ctx.pad_x;
    int unpad_h = segmentation.height - 2 * ctx.pad_y;
    
    cv::Mat unpadded_mask;
    if (ctx.pad_x > 0 || ctx.pad_y > 0) {
        cv::Rect crop_region(ctx.pad_x, ctx.pad_y, unpad_w, unpad_h);
        unpadded_mask = mask_image(crop_region).clone();
    } else {
        unpadded_mask = mask_image;
    }

    cv::Mat resized_mask;
    cv::resize(unpadded_mask, resized_mask, cv::Size(frame.cols, frame.rows), 
               0, 0, cv::INTER_NEAREST);

    // Blend with original image
    cv::Mat result;
    cv::addWeighted(frame, 1.0 - alpha, resized_mask, alpha, 0, result);

    return result;
}

/**
 * @brief Draw classification result on the image
 * @param frame Original image
 * @param classification Classification result
 * @param font_scale Label font scale
 * @return Image with classification label
 */
inline cv::Mat drawClassification(const cv::Mat& frame,
                                  const ClassificationResult& classification,
                                  double font_scale = 1.0) {
    cv::Mat result = frame.clone();

    std::string label = "Class: " + classification.class_name + 
                        " (" + std::to_string(static_cast<int>(classification.confidence * 100)) + "%)";

    // Draw label at top-left corner
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                         font_scale, 2, &baseline);

    cv::rectangle(result, cv::Point(10, 10),
                  cv::Point(20 + text_size.width, 20 + text_size.height + baseline),
                  cv::Scalar(0, 0, 0), -1);

    cv::putText(result, label, cv::Point(15, 15 + text_size.height),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), 2);

    return result;
}

}  // namespace dxapp

#endif  // DXAPP_VISUALIZATION_HPP
