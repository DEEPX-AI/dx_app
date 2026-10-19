/**
 * @file yolopv2_384x640_factory.hpp
 * @brief Yolopv2Factory: panoptic driving perception
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
 *   - Each frame's masks travel in that frame's PanopticResult; the postprocessor
 *     and the visualizer share nothing.
 */

#ifndef YOLOPV2_384X640_FACTORY_HPP
#define YOLOPV2_384X640_FACTORY_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/base/i_factory.hpp"
#include "common/config/model_config.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "yolopv2_postprocess.h"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolopv2_384x640 {

/// Decodes one frame: boxes restored from the letterbox to the original
/// image; masks cropped out of the letterbox and resized to the original
/// image (CV_8UC1, 0/1). Stateless apart from `impl`'s fixed parameters.
inline PanopticResult DecodeYOLOPv2(YOLOPv2PostProcess& impl, const dxrt::TensorPtrs& outputs,
                                    const PreprocessContext& ctx) {
    YOLOPv2Result res = impl.postprocess(outputs);

    const float scale = ctx.scale > 0.0f ? ctx.scale : 1.0f;
    const int orig_h = ctx.original_height > 0 ? ctx.original_height : res.mask_height;
    const int orig_w = ctx.original_width > 0 ? ctx.original_width : res.mask_width;
    const int content_h = std::min(static_cast<int>(std::round(orig_h * scale)), res.mask_height);
    const int content_w = std::min(static_cast<int>(std::round(orig_w * scale)), res.mask_width);
    const int top = std::max(0, std::min(ctx.pad_y, res.mask_height - content_h));
    const int left = std::max(0, std::min(ctx.pad_x, res.mask_width - content_w));

    auto convert_mask = [&](const std::vector<int>& src) -> cv::Mat {
        cv::Mat m_int(res.mask_height, res.mask_width, CV_32SC1, const_cast<int*>(src.data()));
        cv::Mat m_u8;
        m_int.convertTo(m_u8, CV_8UC1);  // 0 -> 0, 1 -> 1
        cv::Mat content = m_u8(cv::Rect(left, top, content_w, content_h)).clone();
        cv::Mat resized;
        cv::resize(content, resized, {orig_w, orig_h}, 0, 0, cv::INTER_NEAREST);
        return resized;
    };

    PanopticResult out;
    out.drivable = convert_mask(res.drivable_mask);
    out.lane = convert_mask(res.lane_mask);

    const float px = static_cast<float>(ctx.pad_x);
    const float py = static_cast<float>(ctx.pad_y);
    const float fw = static_cast<float>(orig_w);
    const float fh = static_cast<float>(orig_h);
    out.detections.reserve(res.detections.size());
    for (const auto& d : res.detections) {
        const float x1 = std::max(0.0f, std::min((d.x1 - px) / scale, fw));
        const float y1 = std::max(0.0f, std::min((d.y1 - py) / scale, fh));
        const float x2 = std::max(0.0f, std::min((d.x2 - px) / scale, fw));
        const float y2 = std::max(0.0f, std::min((d.y2 - py) / scale, fh));
        out.detections.push_back(DetectionResult{{x1, y1, x2, y2}, d.confidence, d.class_id, "vehicle"});
    }
    return out;
}

/// Runner path: one PanopticResult per frame.
class YOLOPv2PanopticPostprocessor : public IPostprocessor<PanopticResult> {
   public:
    YOLOPv2PanopticPostprocessor(int input_w, int input_h, float conf, float nms)
        : impl_(input_w, input_h, conf, nms) {}

    std::vector<PanopticResult> process(const dxrt::TensorPtrs& outputs,
                                        const PreprocessContext& ctx) override {
        return std::vector<PanopticResult>(1, DecodeYOLOPv2(impl_, outputs, ctx));
    }
    std::string getModelName() const override { return "YOLOPv2"; }

   private:
    YOLOPv2PostProcess impl_;
};

/// Boxes only: the sync/async runners' boxes path. The multi-model graph
/// uses createPanopticPostprocessor (PanopticResult) instead.
class YOLOPv2PostprocessorWrapper : public IPostprocessor<DetectionResult> {
   public:
    YOLOPv2PostprocessorWrapper(int input_w, int input_h, float conf, float nms)
        : impl_(input_w, input_h, conf, nms) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        return DecodeYOLOPv2(impl_, outputs, ctx).detections;
    }
    std::string getModelName() const override { return "YOLOPv2"; }

   private:
    YOLOPv2PostProcess impl_;
};

/// Blends `mask_orig` (resized to `out` if needed) into `out` with `color`.
inline void OverlayYOLOPv2Mask(cv::Mat& out, const cv::Mat& mask_orig, const cv::Scalar& color,
                               float alpha) {
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

inline void DrawYOLOPv2Vehicles(cv::Mat& out, const std::vector<DetectionResult>& results) {
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
}

/// Draws each result's own drivable (green) and lane (red) masks, then its boxes.
class YOLOPv2Visualizer : public IVisualizer<PanopticResult> {
   public:
    explicit YOLOPv2Visualizer(float drivable_alpha = 0.4f, float lane_alpha = 0.5f)
        : drivable_alpha_(drivable_alpha), lane_alpha_(lane_alpha) {}

    cv::Mat draw(const cv::Mat& frame, const std::vector<PanopticResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat out = frame.clone();
        for (const auto& r : results) {
            if (!r.drivable.empty()) OverlayYOLOPv2Mask(out, r.drivable, cv::Scalar(0, 180, 0), drivable_alpha_);
            if (!r.lane.empty()) OverlayYOLOPv2Mask(out, r.lane, cv::Scalar(0, 0, 200), lane_alpha_);
            DrawYOLOPv2Vehicles(out, r.detections);
        }
        return out;
    }
    void setParameters(int /*line_thickness*/ = 2, double /*font_scale*/ = 0.5,
                       float /*alpha*/ = 0.6f) override {}

   private:
    float drivable_alpha_;
    float lane_alpha_;
};

/// Boxes only, for callers that hold DetectionResult (createVisualizer()).
class YOLOPv2BoxVisualizer : public IVisualizer<DetectionResult> {
   public:
    cv::Mat draw(const cv::Mat& frame, const std::vector<DetectionResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat out = frame.clone();
        DrawYOLOPv2Vehicles(out, results);
        return out;
    }
    void setParameters(int /*line_thickness*/ = 2, double /*font_scale*/ = 0.5,
                       float /*alpha*/ = 0.6f) override {}
};

class Yolopv2Factory : public IPanopticDrivingFactory {
   public:

    Yolopv2Factory(float conf_threshold = 0.25f, float nms_threshold = 0.45f)
        : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }
    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool /*is_ort_configured*/ = false) override {
        return std::make_unique<YOLOPv2PostprocessorWrapper>(
            input_width, input_height, conf_threshold_, nms_threshold_);
    }
    PostprocessorPtr<PanopticResult> createPanopticPostprocessor(int input_width,
                                                                 int input_height) override {
        return std::make_unique<YOLOPv2PanopticPostprocessor>(
            input_width, input_height, conf_threshold_, nms_threshold_);
    }
    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<YOLOPv2BoxVisualizer>();
    }
    VisualizerPtr<PanopticResult> createPanopticVisualizer() override {
        return std::make_unique<YOLOPv2Visualizer>();
    }
    void loadConfig(const dxapp::ModelConfig& config) override {
        conf_threshold_ = config.get<float>("conf_threshold", conf_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override {
        return "yolopv2_384x640";
    }
    std::string getTaskType() const override { return "panoptic_driving_perception"; }

   private:
    float conf_threshold_;
    float nms_threshold_;

private:
};

}  // namespace v_yolopv2_384x640
}  // namespace dxapp

#endif  // YOLOPV2_384X640_FACTORY_HPP
