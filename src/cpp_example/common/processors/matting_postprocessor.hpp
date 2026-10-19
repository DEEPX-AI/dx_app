/**
 * @file matting_postprocessor.hpp
 * @brief PP-Matting postprocessor: a continuous alpha matte, not a class map
 *
 * PaddleSeg's matting models emit ONE channel -- the foreground opacity of every
 * pixel. Neither ppmatting .dxnn is published (both URLs 403), so this is written to
 * that output shape and refuses anything else.
 *
 * The result carries both forms on purpose. `mask` is the alpha thresholded into a
 * 0/1 class map, because SemanticSegmentationVisualizer reads `mask` as class IDs and
 * would otherwise render a continuous alpha through a class palette; `alpha` is the
 * matte itself, for anything that wants to composite with it.
 *
 * A multi-class segmentation head is refused rather than sliced: taking channel 0 of a
 * 19-class Cityscapes output would produce a believable image and be wrong.
 *
 * Mirrors src/python_example/common/processors/matting_postprocessor.py, whose unit
 * tests pin this behaviour.
 */

#ifndef MATTING_POSTPROCESSOR_HPP
#define MATTING_POSTPROCESSOR_HPP

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "common/base/i_processor.hpp"
#include "common/processors/postprocess_utils.hpp"

namespace dxapp {

class PPMattingPostprocessor : public IPostprocessor<SegmentationResult> {
public:
    PPMattingPostprocessor(int input_width = 512, int input_height = 512,
                           float alpha_threshold = 0.5f)
        : input_width_(input_width), input_height_(input_height),
          alpha_threshold_(alpha_threshold) {}

    std::vector<SegmentationResult> process(const dxrt::TensorPtrs& outputs,
                                            const PreprocessContext& ctx) override {
        cv::Mat alpha = findAlpha(outputs);

        // A model compiled without the final sigmoid emits logits; squash those and
        // leave a map that is already a probability alone.
        double lo = 0.0, hi = 0.0;
        cv::minMaxLoc(alpha, &lo, &hi);
        if (lo < 0.0 || hi > 1.0) {
            cv::Mat negated;
            cv::exp(-alpha, negated);
            alpha = 1.0 / (1.0 + negated);
        }

        if (alpha.cols != ctx.original_width || alpha.rows != ctx.original_height) {
            cv::resize(alpha, alpha,
                       cv::Size(ctx.original_width, ctx.original_height), 0, 0,
                       cv::INTER_LINEAR);
        }
        cv::min(cv::max(alpha, 0.0f), 1.0f, alpha);

        SegmentationResult result;
        result.width = ctx.original_width;
        result.height = ctx.original_height;
        result.class_ids = {0, 1};
        result.class_names = {"background", "foreground"};
        result.alpha = alpha;
        result.mask.resize(static_cast<size_t>(alpha.rows) * alpha.cols);
        for (int y = 0; y < alpha.rows; ++y) {
            const float* row = alpha.ptr<float>(y);
            for (int x = 0; x < alpha.cols; ++x) {
                result.mask[static_cast<size_t>(y) * alpha.cols + x] =
                    row[x] > alpha_threshold_ ? 1 : 0;
            }
        }
        return {std::move(result)};
    }

    std::string getModelName() const override { return "PP-Matting"; }

private:
    /// The HxW alpha map, in whatever single-channel layout it arrives in.
    cv::Mat findAlpha(const dxrt::TensorPtrs& outputs) const {
        for (const auto& o : outputs) {
            const auto& s = o->shape();
            auto* data = const_cast<float*>(static_cast<const float*>(o->data()));
            if (s.size() == 4 && s[0] == 1) {
                if (s[1] == 1) {                                  // NCHW
                    return cv::Mat(static_cast<int>(s[2]), static_cast<int>(s[3]),
                                   CV_32FC1, data).clone();
                }
                if (s[3] == 1) {                                  // NHWC
                    return cv::Mat(static_cast<int>(s[1]), static_cast<int>(s[2]),
                                   CV_32FC1, data).clone();
                }
            } else if (s.size() == 3 && s[0] == 1) {
                return cv::Mat(static_cast<int>(s[1]), static_cast<int>(s[2]),
                               CV_32FC1, data).clone();
            } else if (s.size() == 2) {
                return cv::Mat(static_cast<int>(s[0]), static_cast<int>(s[1]),
                               CV_32FC1, data).clone();
            }
        }
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] PPMattingPostprocessor - no single-channel alpha map "
               "found in the model outputs.\n"
            << postprocess_utils::format_tensor_shapes(outputs)
            << "  A matting model emits one channel: (1, 1, H, W), (1, H, W, 1) or "
               "(1, H, W). A multi-class segmentation head is refused here on purpose "
               "-- silently taking its channel 0 would yield a believable but wrong "
               "matte. Use a semantic-segmentation postprocessor for a class map.\n";
        throw PostprocessConfigError(msg.str());
    }

    int input_width_;
    int input_height_;
    float alpha_threshold_;
};

}  // namespace dxapp

#endif  // MATTING_POSTPROCESSOR_HPP
