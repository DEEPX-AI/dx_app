/**
 * @file superpoint_factory.hpp
 * @brief SuperPointFactory: self-supervised interest point detector and descriptor
 *
 * Input:  UINT8 [1, 480, 640, 1] — grayscale
 * Outputs:
 *   semi: FLOAT [1, 65, Hc, Wc]  — heatmap
 *   desc: FLOAT [1, 256, Hc, Wc] — descriptors
 */

#ifndef SUPERPOINT_FACTORY_HPP
#define SUPERPOINT_FACTORY_HPP

#include <memory>

#include "common/base/i_factory.hpp"
#include "common/processors/grayscale_preprocessor.hpp"
#include "common/processors/pose_postprocessor.hpp"
#include "common/visualizers/pose_visualizer.hpp"
#include "common/config/model_config.hpp"
#include "superpoint_postprocess.h"
#include "superpoint_tracker.hpp"

namespace dxapp {

/**
 * @brief Wraps SuperPointPostProcess to produce PoseResult (keypoints only)
 *        and keeps a shared SuperPointTracker up to date with every frame.
 */
class SuperPointPostprocessorWrapper : public IPostprocessor<PoseResult> {
public:
    SuperPointPostprocessorWrapper(std::shared_ptr<SuperPointTracker> tracker,
                                   int input_w, int input_h,
                                   float conf_threshold = 0.015f,
                                   int top_k = 500)
        : tracker_(std::move(tracker)),
          impl_(input_w, input_h, conf_threshold, top_k) {}

    std::vector<PoseResult> process(const dxrt::TensorPtrs& outputs,
                                    const PreprocessContext& ctx) override {
        SuperPointResult sp = impl_.postprocess(outputs);

        PoseResult pose;
        pose.confidence = 1.0f;
        pose.keypoints.reserve(sp.keypoints.size());
        for (const auto& kp : sp.keypoints)
            pose.keypoints.emplace_back(kp.x, kp.y, kp.score);

        std::vector<PoseResult> results = {std::move(pose)};
        detail::scalePoseResults(results, ctx);

        // Feed tracker with scaled keypoints + raw descriptors
        if (tracker_) {
            const std::vector<Keypoint>& kps = results[0].keypoints;
            std::vector<float> xs, ys;
            xs.reserve(kps.size());
            ys.reserve(kps.size());
            for (const auto& kp : kps) { xs.push_back(kp.x); ys.push_back(kp.y); }
            tracker_->update(xs, ys, sp.descriptors);
        }

        return results;
    }

    std::string getModelName() const override { return "SuperPoint"; }

private:
    std::shared_ptr<SuperPointTracker> tracker_;
    SuperPointPostProcess impl_;
};

/**
 * @brief Draws keypoints and inter-frame tracking lines from the shared tracker.
 *
 * Tracking lines are coloured using a jet colormap that encodes matching
 * quality (blue = low confidence, red = high confidence).
 */
class SuperPointTrackingVisualizer : public IVisualizer<PoseResult> {
public:
    explicit SuperPointTrackingVisualizer(std::shared_ptr<SuperPointTracker> tracker,
                                          int radius = 2,
                                          float conf_threshold = 0.015f,
                                          int min_track_length = 2)
        : tracker_(std::move(tracker)),
          radius_(radius),
          conf_threshold_(conf_threshold),
          min_track_length_(min_track_length) {}

    cv::Mat draw(const cv::Mat& image,
                 const std::vector<PoseResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat output = image.clone();

        // Scale radius for consistent visual appearance across image sizes.
        // Reference: 960×540 (diagonal ≈ 1100 px). Images smaller than the
        // reference get a proportionally smaller radius so circles don't appear
        // oversized when the display window scales the image up.
        const double ref_diag = std::sqrt(960.0 * 960.0 + 540.0 * 540.0);
        const double img_diag = std::sqrt(
            static_cast<double>(output.cols) * output.cols +
            static_cast<double>(output.rows) * output.rows);
        const double r_scale = std::min(1.0, img_diag / ref_diag);
        const int adj_radius = std::max(1, static_cast<int>(std::round(radius_ * r_scale)));
        const int adj_max    = std::max(adj_radius, static_cast<int>(std::round((radius_ + 2) * r_scale)));

        // Draw keypoints
        int total = 0;
        for (const auto& pose : results) {
            for (const auto& kp : pose.keypoints) {
                if (kp.confidence < conf_threshold_) continue;
                int r = std::max(1, static_cast<int>(adj_radius * kp.confidence * 3));
                r = std::min(r, adj_max);
                cv::circle(output,
                           cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                           r, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                ++total;
            }
        }

        // Draw inter-frame tracking lines
        if (tracker_)
            tracker_->drawTracks(output, min_track_length_);

        cv::putText(output,
                    "Keypoints: " + std::to_string(total),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2);
        return output;
    }

    void setParameters(int /*line_thickness*/ = 2,
                       double /*font_scale*/ = 0.5,
                       float /*alpha*/ = 0.6f) override {}

private:
    std::shared_ptr<SuperPointTracker> tracker_;
    int radius_;
    float conf_threshold_;
    int min_track_length_;
};

class SuperPointFactory : public IPoseFactory {
public:
    SuperPointFactory(float conf_threshold = 0.015f, int top_k = 500,
                      float nn_thresh = 0.7f, int track_max_length = 5,
                      int min_track_length = 2, float max_pixel_dist = 100.f)
        : conf_threshold_(conf_threshold),
          top_k_(top_k),
          nn_thresh_(nn_thresh),
          track_max_length_(track_max_length),
          min_track_length_(min_track_length),
          max_pixel_dist_(max_pixel_dist),
          tracker_(std::make_shared<SuperPointTracker>(track_max_length, nn_thresh, max_pixel_dist)) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<GrayscaleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool /*is_ort_configured*/ = false) override {
        return std::make_unique<SuperPointPostprocessorWrapper>(
            tracker_, input_width, input_height, conf_threshold_, top_k_);
    }

    VisualizerPtr<PoseResult> createVisualizer() override {
        return std::make_unique<SuperPointTrackingVisualizer>(
            tracker_, /*radius=*/2, conf_threshold_, min_track_length_);
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        conf_threshold_ = config.get<float>("conf_threshold", conf_threshold_);
        top_k_ = config.get<int>("top_k", top_k_);
        nn_thresh_ = config.get<float>("nn_thresh", nn_thresh_);
        min_track_length_ = config.get<int>("min_track_length", min_track_length_);
        max_pixel_dist_ = config.get<float>("max_pixel_dist", max_pixel_dist_);
    }

    std::string getModelName() const override { return "SuperPoint"; }
    std::string getTaskType() const override { return "keypoint_detection"; }

private:
    float conf_threshold_;
    int top_k_;
    float nn_thresh_;
    int track_max_length_;
    int min_track_length_;
    float max_pixel_dist_;
    std::shared_ptr<SuperPointTracker> tracker_;  // shared by postprocessor + visualizer
};

}  // namespace dxapp

#endif  // SUPERPOINT_FACTORY_HPP
