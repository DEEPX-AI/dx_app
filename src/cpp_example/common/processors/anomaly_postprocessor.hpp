/**
 * @file anomaly_postprocessor.hpp
 * @brief Anomaly-detection feature response -- deliberately NOT an anomaly score
 *
 * The four anomaly models in the DX Model Zoo 2_5_0 additions are single networks out
 * of multi-network methods:
 *
 *  * EfficientAD-M ships as three .dxnn files, and its anomaly map is
 *    0.5*norm(mean((teacher - student[:384])^2)) + 0.5*norm(mean((ae - student[384:])^2))
 *    -- it needs all three at once.
 *  * PatchCore needs a memory bank of training-set features to turn its embedding
 *    into a distance.
 *
 * EfficientAD is now served by EfficientADPostprocessor, which the anomaly runner
 * feeds with all three networks. THIS class remains for PATCHCORE, whose real metric
 * needs a memory bank of training-set features -- a fit step, not a model -- so a
 * single network can only report the per-pixel magnitude of its feature response.
 * That is a real, working application; it is not the published PatchCore metric.
 *
 * Mirrors src/python_example/common/processors/anomaly_postprocessor.py, whose unit
 * tests pin this behaviour, and is cross-checked against it numerically.
 */

#ifndef ANOMALY_POSTPROCESSOR_HPP
#define ANOMALY_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "common/base/i_processor.hpp"
#include "common/processors/efficientad_postprocessor.hpp"
#include "common/processors/postprocess_utils.hpp"

namespace dxapp {

class AnomalyFeaturePostprocessor : public IPostprocessor<AnomalyResult> {
public:
    AnomalyFeaturePostprocessor(int input_width = 256, int input_height = 256)
        : input_width_(input_width), input_height_(input_height) {}

    std::vector<AnomalyResult> process(const dxrt::TensorPtrs& outputs,
                                       const PreprocessContext& ctx) override {
        int channels = 0, height = 0, width = 0;
        bool channels_last = false;
        const float* data = findFeatures(outputs, channels, height, width,
                                         channels_last);

        // Per-pixel L2 norm across channels: the magnitude of the feature response,
        // which is what every distance-based anomaly method reduces to per location.
        cv::Mat response(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            float* out = response.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                double sum = 0.0;
                for (int c = 0; c < channels; ++c) {
                    const std::ptrdiff_t idx = channels_last
                        ? ((static_cast<std::ptrdiff_t>(y) * width + x) * channels + c)
                        : ((static_cast<std::ptrdiff_t>(c) * height + y) * width + x);
                    const double v = data[idx];
                    sum += v * v;
                }
                out[x] = static_cast<float>(std::sqrt(sum));
            }
        }

        double lo = 0.0, hi = 0.0;
        cv::minMaxLoc(response, &lo, &hi);
        cv::Mat normalised;
        if (hi - lo > 1e-12) {
            normalised = (response - lo) / (hi - lo);
        } else {
            // A featureless map (all channels constant) has no contrast to normalise.
            // Reporting it as uniformly hot would be the opposite of the truth.
            normalised = cv::Mat::zeros(response.size(), CV_32FC1);
        }

        cv::Mat heatmap;
        cv::resize(normalised, heatmap,
                   cv::Size(ctx.original_width, ctx.original_height), 0, 0,
                   cv::INTER_LINEAR);

        cv::threshold(heatmap, heatmap, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(heatmap, heatmap, 0.0, 0.0, cv::THRESH_TOZERO);

        AnomalyResult result;
        result.heatmap = heatmap;
        // The percentile of the NORMALISED map, as the Python peer reports: with one
        // network there is no disagreement to measure in raw units, so the number is
        // frame-relative and says so on the frame.
        result.score = linearPercentile(
            std::vector<float>(heatmap.begin<float>(), heatmap.end<float>()), 99.0);
        result.channels = channels;
        return {std::move(result)};
    }

    std::string getModelName() const override { return "anomaly_feature"; }

private:
    /**
     * The (1, C, H, W) or (1, H, W, C) feature map.
     *
     * The spatial grid of these models is square -- 64x64 at a 256x256 input, 28x28 at
     * 224x224 -- which is what tells the layouts apart: channels-first has the equal
     * pair LAST, channels-last has it FIRST. Channels-first wins when all three are
     * equal, because that is what DXRT emits for these networks. The Python peer had
     * this test inverted at first, and a wrong answer transposes the heatmap silently.
     */
    const float* findFeatures(const dxrt::TensorPtrs& outputs, int& channels,
                              int& height, int& width, bool& channels_last) const {
        for (const auto& o : outputs) {
            const auto& s = o->shape();
            if (s.size() != 4 || s[0] != 1) continue;
            const int a = static_cast<int>(s[1]);
            const int b = static_cast<int>(s[2]);
            const int c = static_cast<int>(s[3]);
            if (a == b && b != c) {
                channels_last = true;
                channels = c; height = a; width = b;
            } else {
                channels_last = false;
                channels = a; height = b; width = c;
            }
            return static_cast<const float*>(o->data());
        }
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] AnomalyFeaturePostprocessor - no (1, C, H, W) feature "
               "map found in the model outputs.\n"
            << postprocess_utils::format_tensor_shapes(outputs)
            << "  EfficientAD's teacher/student/autoencoder and PatchCore's backbone "
               "all emit a spatial feature map; a flat embedding cannot be turned into "
               "a per-pixel response.\n";
        throw PostprocessConfigError(msg.str());
    }

    int input_width_;
    int input_height_;
};

}  // namespace dxapp

#endif  // ANOMALY_POSTPROCESSOR_HPP
