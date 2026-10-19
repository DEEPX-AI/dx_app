/**
 * @file efficientad_postprocessor.hpp
 * @brief EfficientAD's published anomaly map, from all three of its networks
 *
 * MEASURED on DX-RT 3.5.0, the EfficientAD-M triple at 256x256:
 *
 *     efficientad-m-teacher_256x256      out teacher_features [1, 384, 64, 64]
 *     efficientad-m-student_256x256      out student_features [1, 768, 64, 64]
 *     efficientad-m-autoencoder_256x256  out teacher_features [1, 384, 64, 64]
 *
 * 768 = 384 + 384. The student predicts the TEACHER with its first half and the
 * AUTOENCODER with its second, and EfficientAD scores the two disagreements:
 *
 *     st = mean((teacher     - student[:384])^2, axis=C)
 *     ae = mean((autoencoder - student[384:])^2, axis=C)
 *     anomaly = 0.5*norm(st) + 0.5*norm(ae)
 *
 * This is why a one-network example could not be right: the magnitude of a single
 * feature map is not a disagreement, and a disagreement is the entire method.
 *
 * What is still missing, and why: the reference implementation normalises with
 * q_st/q_ae quantiles fitted on the training set, which is what makes its score
 * comparable across images and against a published MVTec number. A .dxnn carries no
 * such constants, so each branch is normalised against its own frame here.
 *
 * Teacher and autoencoder are both (1, 384, H, W), so nothing in the data
 * distinguishes them. Which output is which is therefore DECLARED: the factory passes
 * the roles in arrival order, because `-m` names the primary network and the
 * companions follow it. Absent roles the class falls back to shape order -- the
 * 768-channel map is the student, the remaining two are teacher then autoencoder as
 * they arrive.
 *
 * Mirrors src/python_example/common/processors/efficientad_postprocessor.py, whose
 * unit tests pin this behaviour, and is cross-checked against it numerically.
 */

#ifndef EFFICIENTAD_POSTPROCESSOR_HPP
#define EFFICIENTAD_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common/base/i_processor.hpp"
#include "common/processors/postprocess_utils.hpp"

namespace dxapp {

/**
 * @brief numpy.percentile with its default linear interpolation.
 *
 * Written out rather than approximated with a single index, because the cross-tree
 * comparison against the Python peer is exact to float precision and a nearest-rank
 * percentile drifts from numpy by a whole order statistic.
 */
inline float linearPercentile(std::vector<float> values, double q) {
    if (values.empty()) return 0.0f;
    const double pos = (q / 100.0) * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    std::nth_element(values.begin(), values.begin() + lo, values.end());
    const double low = values[lo];
    if (hi == lo) return static_cast<float>(low);
    const double high = *std::min_element(values.begin() + lo + 1, values.end());
    return static_cast<float>(low + (high - low) * (pos - static_cast<double>(lo)));
}

class EfficientADPostprocessor : public IPostprocessor<AnomalyResult> {
public:
    EfficientADPostprocessor(int input_width = 256, int input_height = 256,
                             std::vector<std::string> roles = {})
        : input_width_(input_width), input_height_(input_height),
          roles_(std::move(roles)) {}

    std::vector<AnomalyResult> process(const dxrt::TensorPtrs& outputs,
                                       const PreprocessContext& ctx) override {
        if (outputs.size() != 3) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] EfficientADPostprocessor - needs all three "
                   "EfficientAD networks, got " << outputs.size() << " output(s).\n"
                << postprocess_utils::format_tensor_shapes(outputs)
                << "  Expected a 768-channel student plus a 384-channel teacher and a "
                   "384-channel autoencoder, in that order.\n"
                   "  The student alone cannot be scored: 768 = 384 (student-teacher "
                   "branch) + 384 (autoencoder branch), and both halves need their "
                   "target to be a disagreement at all.\n";
            throw PostprocessConfigError(msg.str());
        }

        std::vector<Feature> maps;
        maps.reserve(outputs.size());
        for (const auto& o : outputs) maps.push_back(toFeature(o, outputs));

        const Feature* student = nullptr;
        const Feature* teacher = nullptr;
        const Feature* autoenc = nullptr;
        if (roles_.size() == maps.size()) {
            for (std::size_t i = 0; i < maps.size(); ++i) {
                if (roles_[i] == "student") student = &maps[i];
                else if (roles_[i] == "teacher") teacher = &maps[i];
                else if (roles_[i] == "autoencoder") autoenc = &maps[i];
            }
            if (student == nullptr || teacher == nullptr || autoenc == nullptr) {
                std::ostringstream msg;
                msg << "[DXAPP] [ERROR] EfficientADPostprocessor - the declared roles "
                       "do not name every network:";
                for (const auto& r : roles_) msg << " " << r;
                msg << "\n";
                throw PostprocessConfigError(msg.str());
            }
        } else {
            std::size_t widest = 0;
            for (std::size_t i = 1; i < maps.size(); ++i) {
                if (maps[i].channels > maps[widest].channels) widest = i;
            }
            student = &maps[widest];
            for (std::size_t i = 0; i < maps.size(); ++i) {
                if (i == widest) continue;
                if (teacher == nullptr) teacher = &maps[i];
                else autoenc = &maps[i];
            }
        }

        const int half = teacher->channels;
        if (student->channels != teacher->channels + autoenc->channels
                || teacher->channels != autoenc->channels) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] EfficientADPostprocessor - the student's channels "
                   "do not split into its two targets.\n"
                << "  student " << student->channels << ", teacher "
                << teacher->channels << ", autoencoder " << autoenc->channels << "\n"
                << "  EfficientAD's student emits exactly twice its teacher's "
                   "channels.\n";
            throw PostprocessConfigError(msg.str());
        }

        const int h = teacher->height, w = teacher->width;
        cv::Mat st(h, w, CV_32FC1), ae(h, w, CV_32FC1);
        for (int y = 0; y < h; ++y) {
            float* st_row = st.ptr<float>(y);
            float* ae_row = ae.ptr<float>(y);
            for (int x = 0; x < w; ++x) {
                double sum_st = 0.0, sum_ae = 0.0;
                for (int c = 0; c < half; ++c) {
                    const double d_st = teacher->at(c, y, x) - student->at(c, y, x);
                    const double d_ae = autoenc->at(c, y, x)
                                        - student->at(c + half, y, x);
                    sum_st += d_st * d_st;
                    sum_ae += d_ae * d_ae;
                }
                st_row[x] = static_cast<float>(sum_st / half);
                ae_row[x] = static_cast<float>(sum_ae / half);
            }
        }

        // Two maps, two jobs. The HEATMAP is normalised against its own frame, which
        // is what makes a hot region visible whatever the absolute magnitudes are.
        // The SCORE is read off the RAW disagreement, because a per-frame normalised
        // map has roughly the same 99th percentile whether the anomaly is faint or
        // glaring -- it could not rank one frame against another.
        cv::Mat combined = 0.5f * normalise(st) + 0.5f * normalise(ae);
        cv::Mat raw = 0.5f * st + 0.5f * ae;

        cv::Mat heatmap;
        cv::resize(combined, heatmap,
                   cv::Size(ctx.original_width, ctx.original_height), 0, 0,
                   cv::INTER_LINEAR);
        cv::threshold(heatmap, heatmap, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(heatmap, heatmap, 0.0, 0.0, cv::THRESH_TOZERO);

        AnomalyResult result;
        result.heatmap = heatmap;
        // Comparable across frames for THIS model set, and to nothing else: the
        // published EfficientAD score divides by q_st/q_ae quantiles fitted on the
        // training set, and a .dxnn carries no such constants.
        result.score = linearPercentile(
            std::vector<float>(raw.begin<float>(), raw.end<float>()), 99.0);
        result.channels = student->channels;
        return {std::move(result)};
    }

    std::string getModelName() const override { return "efficientad"; }

private:
    /// A feature map with its layout resolved, indexable as (c, y, x).
    struct Feature {
        const float* data{nullptr};
        int channels{0};
        int height{0};
        int width{0};
        bool channels_last{false};

        float at(int c, int y, int x) const {
            const std::ptrdiff_t idx = channels_last
                ? ((static_cast<std::ptrdiff_t>(y) * width + x) * channels + c)
                : ((static_cast<std::ptrdiff_t>(c) * height + y) * width + x);
            return data[idx];
        }
    };

    /**
     * The spatial grid of these networks is square (64x64 at a 256x256 input), which
     * is what tells the layouts apart: in NHWC the equal pair is FIRST, in NCHW it is
     * LAST. A wrong guess here transposes the map without erroring.
     */
    Feature toFeature(const dxrt::TensorPtr& o,
                      const dxrt::TensorPtrs& all) const {
        const auto& s = o->shape();
        if (s.size() != 4 || s[0] != 1) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] EfficientADPostprocessor - expected a "
                   "(1, C, H, W) feature map.\n"
                << postprocess_utils::format_tensor_shapes(all);
            throw PostprocessConfigError(msg.str());
        }
        const int a = static_cast<int>(s[1]);
        const int b = static_cast<int>(s[2]);
        const int c = static_cast<int>(s[3]);
        Feature f;
        f.data = static_cast<const float*>(o->data());
        if (a == b && b != c) {
            f.channels_last = true;
            f.channels = c; f.height = a; f.width = b;
        } else {
            f.channels_last = false;
            f.channels = a; f.height = b; f.width = c;
        }
        return f;
    }

    /// Frame-relative 0..1. A flat branch stays at 0 rather than becoming all-hot.
    static cv::Mat normalise(const cv::Mat& response) {
        double lo = 0.0, hi = 0.0;
        cv::minMaxLoc(response, &lo, &hi);
        if (hi - lo > 1e-12) return (response - lo) / (hi - lo);
        return cv::Mat::zeros(response.size(), CV_32FC1);
    }

    int input_width_;
    int input_height_;
    std::vector<std::string> roles_;
};

}  // namespace dxapp

#endif  // EFFICIENTAD_POSTPROCESSOR_HPP
