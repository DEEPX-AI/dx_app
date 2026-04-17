/**
 * @file segmentation_visualizer.hpp
 * @brief Semantic and instance segmentation result visualizers
 */

#ifndef SEGMENTATION_VISUALIZER_HPP
#define SEGMENTATION_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"

namespace dxapp {

// Cityscapes-like color palette for segmentation
static const std::vector<cv::Vec3b> SEGMENTATION_COLORS = {
    {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156},
    {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0},
    {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
    {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100},
    {0, 80, 100}, {0, 0, 230}, {119, 11, 32}, {0, 0, 0}
};

/**
 * @brief Visualizer for semantic segmentation results
 */
class SemanticSegmentationVisualizer : public IVisualizer<SegmentationResult> {
public:
    SemanticSegmentationVisualizer() = default;

    /**
     * @brief Construct with a custom color palette.
     * @param palette  Custom color palette (index = class id).
     * @param skip_bg  When true, class 0 is treated as background and not drawn.
     */
    SemanticSegmentationVisualizer(std::vector<cv::Vec3b> palette, bool skip_bg = true)
        : custom_palette_(std::move(palette)), skip_background_(skip_bg) {}

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<SegmentationResult>& results,
                 const PreprocessContext& ctx) override {
        if (results.empty()) return frame.clone();
        
        const auto& seg = results[0];  // Typically one result for semantic seg
        const auto& palette = custom_palette_.empty() ? SEGMENTATION_COLORS : custom_palette_;
        
        // Create colored mask
        cv::Mat colored_mask(seg.height, seg.width, CV_8UC3);
        auto* dst = colored_mask.ptr<cv::Vec3b>();
        int total = seg.height * seg.width;
        int psize = static_cast<int>(palette.size());
        for (int i = 0; i < total; ++i) {
            int class_id = seg.mask[i];
            const auto& c = palette[class_id % psize];
            if (skip_background_ && c == cv::Vec3b(0, 0, 0))
                dst[i] = cv::Vec3b(0, 0, 0);
            else
                dst[i] = c;
        }
        
        // Remove letterbox padding before resize (matching original)
        cv::Mat unpadded_mask;
        if (ctx.pad_x > 0 || ctx.pad_y > 0) {
            int unpad_w = seg.width - 2 * ctx.pad_x;
            int unpad_h = seg.height - 2 * ctx.pad_y;
            // Safety check to prevent ROI assertion errors
            if (unpad_w > 0 && unpad_h > 0 &&
                ctx.pad_x + unpad_w <= colored_mask.cols &&
                ctx.pad_y + unpad_h <= colored_mask.rows) {
                cv::Rect crop_region(ctx.pad_x, ctx.pad_y, unpad_w, unpad_h);
                unpadded_mask = colored_mask(crop_region).clone();
            } else {
                unpadded_mask = colored_mask;
            }
        } else {
            unpadded_mask = colored_mask;
        }
        
        // Resize mask to match frame size. Use linear interpolation to reduce
        // blocky appearance when upscaling low-res segmentation outputs.
        cv::Mat resized_mask;
        cv::resize(unpadded_mask, resized_mask, frame.size(), 0, 0, cv::INTER_LINEAR);
        
        // Blend with original frame
        cv::Mat output;
        cv::addWeighted(frame, 1.0 - alpha_, resized_mask, alpha_, 0, output);
        
        return output;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        (void)line_thickness;
        (void)font_scale;
        alpha_ = alpha;
    }

private:
    float alpha_{0.6f};
    std::vector<cv::Vec3b> custom_palette_;
    bool skip_background_{false};
};

/**
 * @brief Visualizer for instance segmentation results
 */
class InstanceSegmentationVisualizer : public IVisualizer<InstanceSegmentationResult> {
public:
    InstanceSegmentationVisualizer(bool show_boxes = true) : show_boxes_(show_boxes) {}

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<InstanceSegmentationResult>& results,
                 const PreprocessContext& ctx) override {
        cv::Mat output = frame.clone();

        // Scale factor from original image space to display frame space.
        // ctx.original_width/height reflect the source image dimensions that the
        // postprocessor used when mapping boxes back from model space.  When
        // displayResize() has down-scaled the frame (e.g. 1920x1080 → 960x540)
        // the box coordinates must be scaled accordingly before drawing.
        float disp_scale = 1.0f;
        if (ctx.original_width > 0 && ctx.original_height > 0 &&
            (ctx.original_width > output.cols || ctx.original_height > output.rows)) {
            disp_scale = std::min(static_cast<float>(output.cols) / ctx.original_width,
                                  static_cast<float>(output.rows) / ctx.original_height);
        }
        const float x_off = (ctx.original_width > 0)
            ? (output.cols - ctx.original_width * disp_scale) / 2.0f : 0.0f;
        const float y_off = (ctx.original_height > 0)
            ? (output.rows - ctx.original_height * disp_scale) / 2.0f : 0.0f;

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& inst = results[i];
            
            // Get color for this instance (use instance index for variety)
            cv::Vec3b color = SEGMENTATION_COLORS[i % SEGMENTATION_COLORS.size()];
            cv::Scalar box_color(color[0], color[1], color[2]);
            
            // Draw mask overlay first (so boxes appear on top)
            if (!inst.mask.empty()) {
                cv::Mat binary_mask = convertToBinaryMask(inst.mask);

                if (binary_mask.size() != frame.size()) {
                    cv::resize(binary_mask, binary_mask, frame.size());
                }

                blendMaskRegion(binary_mask, color, alpha_, output);
            }

            // Draw bounding box and label
            if (show_boxes_ && inst.box.size() >= 4) {
                cv::Point pt1(static_cast<int>(inst.box[0] * disp_scale + x_off), static_cast<int>(inst.box[1] * disp_scale + y_off));
                cv::Point pt2(static_cast<int>(inst.box[2] * disp_scale + x_off), static_cast<int>(inst.box[3] * disp_scale + y_off));
                cv::rectangle(output, pt1, pt2, box_color, line_thickness_);
                
                std::string label = inst.class_name + ": " + 
                    std::to_string(static_cast<int>(inst.confidence * 100)) + "%";
                int baseline;
                cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale_, 1, &baseline);
                cv::putText(output, label, cv::Point(pt1.x, std::max(pt1.y - 5, 15)),
                           cv::FONT_HERSHEY_SIMPLEX, font_scale_, box_color, 1);
            }
        }

        return output;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        line_thickness_ = line_thickness;
        font_scale_ = font_scale;
        alpha_ = alpha;
    }

private:
    int line_thickness_{2};
    double font_scale_{0.5};
    float alpha_{0.4f};
    bool show_boxes_{true};

    /** Convert mask to binary uint8 format. */
    static cv::Mat convertToBinaryMask(const cv::Mat& mask) {
        cv::Mat binary_mask;
        if (mask.type() == CV_32FC1 || mask.type() == CV_64FC1) {
            mask.convertTo(binary_mask, CV_8UC1, 255.0);
            cv::threshold(binary_mask, binary_mask, 127, 255, cv::THRESH_BINARY);
        } else {
            mask.convertTo(binary_mask, CV_8UC1);
        }
        return binary_mask;
    }

    /** Blend a color into target where mask > 0. */
    static void blendMaskRegion(const cv::Mat& mask, const cv::Vec3b& color,
                                float alpha, cv::Mat& target) {
        cv::Mat color_mat(target.size(), CV_8UC3, cv::Scalar(color[0], color[1], color[2]));
        cv::Mat blended;
        cv::addWeighted(target, 1.0 - alpha, color_mat, alpha, 0, blended);
        blended.copyTo(target, mask);
    }
};

}  // namespace dxapp

#endif  // SEGMENTATION_VISUALIZER_HPP
