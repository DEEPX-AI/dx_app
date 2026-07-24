/**
 * @file yolopv2_factory.hpp
 * @brief YOLOPv2Factory: panoptic driving perception
 *        (vehicle detection + drivable area + lane line)
 *
 * Outputs:
 *   det0/1/2: [1,255,H,W] YOLO detection heads
 *   output_4: [1,2,H,W]   drivable area segmentation
 *   output_5: [1,1,H,W]   lane line segmentation
 *
 * Design:
 *   - Only detects vehicles (COCO class 3), matching custom_ops.py vehicle_class_index=3.
 *   - Class-agnostic NMS.
 *   - Letterbox-aware coordinate restoration via PreprocessContext.
 *   - Shared YOLOPv2State lets the postprocessor pass masks to the visualizer.
 */

#ifndef YOLOPV2_FACTORY_HPP
#define YOLOPV2_FACTORY_HPP

#include <algorithm>
#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "common/base/i_factory.hpp"
#include "common/config/model_config.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "yolopv2_postprocess.h"

namespace dxapp {

// ---------------------------------------------------------------------------
// Shared state: passes masks from postprocessor to visualizer
// ---------------------------------------------------------------------------
struct YOLOPv2State {
    std::mutex mtx;
    cv::Mat drivable;  // CV_8UC1, original image resolution (after letterbox crop+resize)
    cv::Mat lane;      // CV_8UC1, original image resolution
};

// ---------------------------------------------------------------------------
// Postprocessor wrapper
// ---------------------------------------------------------------------------
class YOLOPv2PostprocessorWrapper : public IPostprocessor<DetectionResult> {
   public:
    YOLOPv2PostprocessorWrapper(int input_w, int input_h, float conf, float nms,
                                 std::shared_ptr<YOLOPv2State> state)
        : impl_(input_w, input_h, conf, nms), state_(std::move(state)) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                          const PreprocessContext& ctx) override {
        YOLOPv2Result res = impl_.postprocess(outputs);

        // Letterbox parameters for coordinate + mask restoration
        const float scale = ctx.scale > 0.0f ? ctx.scale : 1.0f;
        const int orig_h = ctx.original_height > 0 ? ctx.original_height : res.mask_height;
        const int orig_w = ctx.original_width > 0 ? ctx.original_width : res.mask_width;
        // Content region in model-output space
        const int content_h = std::min(
            static_cast<int>(std::round(orig_h * scale)), res.mask_height);
        const int content_w = std::min(
            static_cast<int>(std::round(orig_w * scale)), res.mask_width);
        const int top = std::max(0, std::min(ctx.pad_y, res.mask_height - content_h));
        const int left = std::max(0, std::min(ctx.pad_x, res.mask_width - content_w));

        // Convert int mask → uint8 Mat, crop letterbox padding, resize to original resolution
        auto convert_mask = [&](const std::vector<int>& src) -> cv::Mat {
            // Build CV_32SC1 Mat from int data (avoids byte-level misinterpretation)
            cv::Mat m_int(res.mask_height, res.mask_width, CV_32SC1,
                          const_cast<int*>(src.data()));
            cv::Mat m_u8;
            m_int.convertTo(m_u8, CV_8UC1);  // 0→0, 1→1
            // Crop the unpadded content region
            cv::Mat content = m_u8(cv::Rect(left, top, content_w, content_h)).clone();
            // Resize to original image resolution
            cv::Mat resized;
            cv::resize(content, resized, {orig_w, orig_h}, 0, 0, cv::INTER_NEAREST);
            return resized;
        };

        {
            std::lock_guard<std::mutex> lk(state_->mtx);
            state_->drivable = convert_mask(res.drivable_mask);
            state_->lane = convert_mask(res.lane_mask);
        }

        // Convert detections, restoring letterbox coords
        const float px = static_cast<float>(ctx.pad_x);
        const float py = static_cast<float>(ctx.pad_y);
        const float fw = static_cast<float>(orig_w);
        const float fh = static_cast<float>(orig_h);

        std::vector<DetectionResult> dets;
        dets.reserve(res.detections.size());
        for (const auto& d : res.detections) {
            const float x1 = std::max(0.0f, std::min((d.x1 - px) / scale, fw));
            const float y1 = std::max(0.0f, std::min((d.y1 - py) / scale, fh));
            const float x2 = std::max(0.0f, std::min((d.x2 - px) / scale, fw));
            const float y2 = std::max(0.0f, std::min((d.y2 - py) / scale, fh));
            dets.push_back(DetectionResult{
                {x1, y1, x2, y2}, d.confidence, d.class_id, "vehicle"});
        }
        return dets;
    }

    std::string getModelName() const override { return "YOLOPv2"; }

   private:
    YOLOPv2PostProcess impl_;
    std::shared_ptr<YOLOPv2State> state_;
};

// ---------------------------------------------------------------------------
// Visualizer: overlays drivable (green) + lane (red) + detection boxes
// ---------------------------------------------------------------------------
class YOLOPv2Visualizer : public IVisualizer<DetectionResult> {
   public:
    explicit YOLOPv2Visualizer(std::shared_ptr<YOLOPv2State> state,
                                float drivable_alpha = 0.4f,
                                float lane_alpha = 0.5f)
        : state_(std::move(state)),
          drivable_alpha_(drivable_alpha),
          lane_alpha_(lane_alpha) {}

    cv::Mat draw(const cv::Mat& frame, const std::vector<DetectionResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat out = frame.clone();

        // Retrieve masks
        cv::Mat drivable_model, lane_model;
        {
            std::lock_guard<std::mutex> lk(state_->mtx);
            drivable_model = state_->drivable.clone();
            lane_model = state_->lane.clone();
        }

        if (!drivable_model.empty()) {
            overlay_mask(out, drivable_model, cv::Scalar(0, 180, 0), drivable_alpha_);
        }
        if (!lane_model.empty()) {
            overlay_mask(out, lane_model, cv::Scalar(0, 0, 200), lane_alpha_);
        }

        // Draw vehicle bounding boxes
        for (const auto& det : results) {
            const int x1 = static_cast<int>(det.box[0]);
            const int y1 = static_cast<int>(det.box[1]);
            const int x2 = static_cast<int>(det.box[2]);
            const int y2 = static_cast<int>(det.box[3]);
            cv::rectangle(out, {x1, y1}, {x2, y2}, cv::Scalar(0, 255, 255), 2);
            const std::string label =
                "vehicle " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            cv::putText(out, label, {x1, std::max(y1 - 4, 10)}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(0, 0, 0), 2);
            cv::putText(out, label, {x1, std::max(y1 - 4, 10)}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(0, 255, 255), 1);
        }
        return out;
    }

    void setParameters(int /*line_thickness*/ = 2, double /*font_scale*/ = 0.5,
                       float /*alpha*/ = 0.6f) override {}

   private:
    void overlay_mask(cv::Mat& out, const cv::Mat& mask_orig, const cv::Scalar& color,
                      float alpha) {
        // mask_orig is already at original image resolution; resize only if needed
        cv::Mat mask_display;
        if (mask_orig.rows != out.rows || mask_orig.cols != out.cols) {
            cv::resize(mask_orig, mask_display, {out.cols, out.rows}, 0, 0, cv::INTER_NEAREST);
        } else {
            mask_display = mask_orig;
        }
        cv::Mat overlay = out.clone();
        overlay.setTo(color, mask_display > 0);
        cv::addWeighted(out, 1.0f - alpha, overlay, alpha, 0.0, out);
    }

    std::shared_ptr<YOLOPv2State> state_;
    float drivable_alpha_;
    float lane_alpha_;
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
class YOLOPv2Factory : public IPanopticDrivingFactory {
   public:
    YOLOPv2Factory(float conf_threshold = 0.25f, float nms_threshold = 0.45f)
        : conf_threshold_(conf_threshold),
          nms_threshold_(nms_threshold),
          state_(std::make_shared<YOLOPv2State>()) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool /*is_ort_configured*/ = false) override {
        return std::make_unique<YOLOPv2PostprocessorWrapper>(
            input_width, input_height, conf_threshold_, nms_threshold_, state_);
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<YOLOPv2Visualizer>(state_);
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        conf_threshold_ = config.get<float>("conf_threshold", conf_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override { return "YOLOPv2"; }
    std::string getTaskType() const override { return "panoptic_driving_perception"; }

   private:
    float conf_threshold_;
    float nms_threshold_;
    std::shared_ptr<YOLOPv2State> state_;
};

}  // namespace dxapp

#endif  // YOLOPV2_FACTORY_HPP
