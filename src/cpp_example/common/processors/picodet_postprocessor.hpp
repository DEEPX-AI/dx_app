/**
 * @file picodet_postprocessor.hpp
 * @brief PP-PicoDet postprocessor: a GFL head delivered one tensor per pyramid level.
 *
 * PicoDet and NanoDet-Plus share the head (anchor-free class scores plus a Distribution
 * Focal Loss box regression), but not the packaging: NanoDet emits ONE concatenated
 * tensor [1, N, num_classes + 4 * (reg_max + 1)], PicoDet a score and a box tensor per
 * level. Measured on DX-RT 3.5.0, pp-shituv1/v2-mainbody-detection_640x640:
 *
 *     [1, 6400, 1] [1, 1600, 1] [1, 400, 1] [1, 100, 1]      class scores, 1 class
 *     [1, 6400, 32] [1, 1600, 32] [1, 400, 32] [1, 100, 32]  box distributions
 *
 * so strides 8/16/32/64 at 640, and 32 = 4 sides x (reg_max + 1), reg_max 7. The scores
 * are already sigmoid; the distributions are not softmaxed. Levels are paired by anchor
 * count, never by tensor order. The same decoding as the Python PicoDetPostprocessor
 * (src/python_example/common/processors/picodet_postprocessor.py).
 */

#ifndef PICODET_POSTPROCESSOR_HPP
#define PICODET_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common_util.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dxapp {

class PicoDetPostprocessor : public IPostprocessor<DetectionResult> {
public:
    PicoDetPostprocessor(int input_width = 640, int input_height = 640,
                         float conf_threshold = 0.3f,
                         float nms_threshold = 0.5f,
                         int num_classes = 1,
                         const std::vector<std::string>& class_names = {})
        : input_width_(input_width), input_height_(input_height),
          conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
          num_classes_(num_classes), class_names_(class_names) {}

    int numClasses() const { return num_classes_; }

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        std::vector<DetectionResult> results;
        std::vector<Level> levels = pairLevels(outputs);

        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        for (std::size_t l = 0; l < levels.size(); ++l) {
            const Level& level = levels[l];
            const int bins = level.box_cols / 4;
            const int grid = level.grid;
            const float stride = static_cast<float>(input_width_) / static_cast<float>(grid);
            for (int i = 0; i < level.anchors; ++i) {
                const float* row = level.scores + static_cast<std::size_t>(i) * level.score_cols;
                int best_cls = 0;
                float best = row[0];
                for (int c = 1; c < level.score_cols; ++c) {
                    if (row[c] > best) {
                        best = row[c];
                        best_cls = c;
                    }
                }
                if (best < conf_threshold_) continue;
                const float* reg = level.boxes + static_cast<std::size_t>(i) * level.box_cols;
                float d[4];
                for (int side = 0; side < 4; ++side) d[side] = integrate(reg + side * bins, bins) * stride;
                const float cx = (static_cast<float>(i % grid) + 0.5f) * stride;
                const float cy = (static_cast<float>(i / grid) + 0.5f) * stride;
                const float x1 = cx - d[0], y1 = cy - d[1], x2 = cx + d[2], y2 = cy + d[3];
                boxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores.push_back(best);
                class_ids.push_back(best_cls);
            }
        }
        if (boxes.empty()) return results;

        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, keep);

        // Undo the letterbox, clipping as the other detection postprocessors do.
        const float gain = std::max(ctx.scale, 1e-6f);
        const float max_x = static_cast<float>(ctx.original_width - 1);
        const float max_y = static_cast<float>(ctx.original_height - 1);
        for (std::size_t k = 0; k < keep.size(); ++k) {
            const int idx = keep[k];
            const cv::Rect2d& b = boxes[idx];
            DetectionResult det;
            det.box = {clip((static_cast<float>(b.x) - ctx.pad_x) / gain, max_x),
                       clip((static_cast<float>(b.y) - ctx.pad_y) / gain, max_y),
                       clip((static_cast<float>(b.x + b.width) - ctx.pad_x) / gain, max_x),
                       clip((static_cast<float>(b.y + b.height) - ctx.pad_y) / gain, max_y)};
            det.confidence = scores[idx];
            det.class_id = class_ids[idx];
            det.class_name = dxapp::common::resolve_class_name(det.class_id, class_names_);
            results.push_back(det);
        }
        return results;
    }

    std::string getModelName() const override { return "PicoDet"; }

private:
    struct Level {
        const float* scores;
        const float* boxes;
        int anchors;
        int grid;
        int score_cols;
        int box_cols;
    };

    /// [score, box] per pyramid level, paired by anchor count (largest grid first): a
    /// tensor [1, N, C] is a box tensor when C is a multiple of 4 and at least 8, else scores.
    static std::vector<Level> pairLevels(const dxrt::TensorPtrs& outputs) {
        std::map<int, std::pair<const float*, int> > scores;
        std::map<int, std::pair<const float*, int> > boxes;
        for (std::size_t t = 0; t < outputs.size(); ++t) {
            const auto shape = outputs[t]->shape();
            if (shape.size() != 3 || shape[0] != 1) continue;
            const int anchors = static_cast<int>(shape[1]);
            const int cols = static_cast<int>(shape[2]);
            const float* data = static_cast<const float*>(outputs[t]->data());
            if (cols % 4 == 0 && cols >= 8) {
                boxes[anchors] = std::make_pair(data, cols);
            } else {
                scores[anchors] = std::make_pair(data, cols);
            }
        }
        std::vector<Level> levels;
        for (std::map<int, std::pair<const float*, int> >::const_reverse_iterator it = boxes.rbegin();
             it != boxes.rend(); ++it) {
            std::map<int, std::pair<const float*, int> >::const_iterator s = scores.find(it->first);
            if (s == scores.end()) continue;
            const int grid = static_cast<int>(std::lround(std::sqrt(static_cast<double>(it->first))));
            if (grid * grid != it->first) continue;  // not a square grid: not a pyramid level
            Level level;
            level.scores = s->second.first;
            level.score_cols = s->second.second;
            level.boxes = it->second.first;
            level.box_cols = it->second.second;
            level.anchors = it->first;
            level.grid = grid;
            levels.push_back(level);
        }
        return levels;
    }

    /// DFL: softmax over the bins, then the expected bin index (grid units).
    static float integrate(const float* side, int bins) {
        const float top = *std::max_element(side, side + bins);
        float sum = 0.0f, weighted = 0.0f;
        for (int b = 0; b < bins; ++b) {
            const float e = std::exp(side[b] - top);
            sum += e;
            weighted += e * static_cast<float>(b);
        }
        return sum > 0.0f ? weighted / sum : 0.0f;
    }

    static float clip(float v, float hi) { return std::max(0.0f, std::min(v, hi)); }

    int input_width_;
    int input_height_;
    float conf_threshold_;
    float nms_threshold_;
    int num_classes_;
    std::vector<std::string> class_names_;
};

}  // namespace dxapp

#endif  // PICODET_POSTPROCESSOR_HPP
