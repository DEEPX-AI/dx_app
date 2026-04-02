/**
 * @file retinaface_postprocessor.hpp
 * @brief RetinaFace face detection postprocessor with 5-point landmarks
 * 
 * Part of DX-APP v3.0.0 refactoring.
 * Ported from Python retinaface_postprocessor.py.
 * 
 * Anchor-based face detection with multi-scale feature maps.
 * Output: 3 tensors (auto-sorted by last dimension):
 *   - scores:    [1, N, 2]   (bg/face softmax, last_dim=2)
 *   - boxes:     [1, N, 4]   (anchor offsets, last_dim=4)
 *   - landmarks: [1, N, 10]  (5 keypoints × 2, last_dim=10)
 * 
 * Prior boxes generated from strides [8, 16, 32].
 * Variance = [0.1, 0.2] for decoding.
 */

#ifndef RETINAFACE_POSTPROCESSOR_HPP
#define RETINAFACE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace dxapp {

class RetinaFacePostprocessor : public IPostprocessor<FaceDetectionResult> {
public:
    RetinaFacePostprocessor(int input_width = 640, int input_height = 640,
                            float score_threshold = 0.5f,
                            float nms_threshold = 0.4f)
        : input_width_(input_width), input_height_(input_height),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold) {
        generatePriorBoxes();
    }

    std::vector<FaceDetectionResult> process(const dxrt::TensorPtrs& outputs,
                                              const PreprocessContext& ctx) override {
        std::vector<FaceDetectionResult> results;
        if (outputs.size() < 3) return results;

        // Sort tensors by last dimension: 2=scores, 4=boxes, 10=landmarks
        const dxrt::TensorPtr* scores_t = nullptr;
        const dxrt::TensorPtr* boxes_t = nullptr;
        const dxrt::TensorPtr* lmks_t = nullptr;
        for (auto& t : outputs) {
            auto shape = t->shape();
            int last_dim = static_cast<int>(shape.back());
            if (last_dim == 2) scores_t = &t;
            else if (last_dim == 4) boxes_t = &t;
            else if (last_dim == 10) lmks_t = &t;
        }
        if (!scores_t || !boxes_t || !lmks_t) return results;

        auto scores_shape = (*scores_t)->shape();
        int N = static_cast<int>(scores_shape.size() == 3 ? scores_shape[1] : scores_shape[0]);
        N = std::min(N, static_cast<int>(priors_.size()));

        const float* scores_data = static_cast<const float*>((*scores_t)->data());
        const float* boxes_data = static_cast<const float*>((*boxes_t)->data());
        const float* lmks_data = static_cast<const float*>((*lmks_t)->data());

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_indices;

        collectCandidates(N, scores_data, boxes_data, nms_boxes, nms_scores, nms_indices);

        if (nms_boxes.empty()) return results;

        std::vector<int> kept;
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, kept);

        for (int k : kept) {
            int i = nms_indices[k];

            float cx, cy, bw, bh;
            decodePriorBox(i, boxes_data, cx, cy, bw, bh);

            float x1 = (cx - bw * 0.5f) * input_width_;
            float y1 = (cy - bh * 0.5f) * input_height_;
            float x2 = (cx + bw * 0.5f) * input_width_;
            float y2 = (cy + bh * 0.5f) * input_height_;

            // Decode 5 landmarks
            float pcx = priors_[i][0], pcy = priors_[i][1];
            float pw  = priors_[i][2], ph  = priors_[i][3];
            std::vector<Keypoint> landmarks(5);
            for (int j = 0; j < 5; ++j) {
                float lx = (pcx + lmks_data[i * 10 + j * 2]     * variance_[0] * pw) * input_width_;
                float ly = (pcy + lmks_data[i * 10 + j * 2 + 1] * variance_[0] * ph) * input_height_;
                landmarks[j] = Keypoint(lx, ly, 1.0f);
            }

            scaleResultCoords(ctx, x1, y1, x2, y2, landmarks);

            FaceDetectionResult face;
            face.box = {
                std::max(0.0f, std::min(x1, static_cast<float>(ctx.original_width))),
                std::max(0.0f, std::min(y1, static_cast<float>(ctx.original_height))),
                std::max(0.0f, std::min(x2, static_cast<float>(ctx.original_width))),
                std::max(0.0f, std::min(y2, static_cast<float>(ctx.original_height)))
            };
            face.confidence = nms_scores[k];
            face.landmarks = std::move(landmarks);
            results.push_back(face);
        }
        return results;
    }

    std::string getModelName() const override { return "RetinaFace"; }

private:
    // Helper: decode center/size from prior box and regression data
    void decodePriorBox(int i, const float* boxes_data,
                        float& cx, float& cy, float& bw, float& bh) const {
        float pcx = priors_[i][0], pcy = priors_[i][1];
        float pw  = priors_[i][2], ph  = priors_[i][3];
        cx = pcx + boxes_data[i * 4 + 0] * variance_[0] * pw;
        cy = pcy + boxes_data[i * 4 + 1] * variance_[0] * ph;
        bw = pw  * std::exp(boxes_data[i * 4 + 2] * variance_[1]);
        bh = ph  * std::exp(boxes_data[i * 4 + 3] * variance_[1]);
    }

    // Helper: filter candidates above threshold and convert to NMS input
    void collectCandidates(int N, const float* scores_data, const float* boxes_data,
                           std::vector<cv::Rect>& nms_boxes,
                           std::vector<float>& nms_scores,
                           std::vector<int>& nms_indices) const {
        for (int i = 0; i < N; ++i) {
            float face_score = scores_data[i * 2 + 1];
            if (face_score < score_threshold_) continue;

            float cx, cy, bw, bh;
            decodePriorBox(i, boxes_data, cx, cy, bw, bh);

            float x1 = (cx - bw * 0.5f) * input_width_;
            float y1 = (cy - bh * 0.5f) * input_height_;
            float x2 = (cx + bw * 0.5f) * input_width_;
            float y2 = (cy + bh * 0.5f) * input_height_;

            nms_indices.push_back(i);
            nms_scores.push_back(face_score);
            nms_boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
        }
    }

    // Helper: scale box corners and landmarks from input space to original image space
    void scaleResultCoords(const PreprocessContext& ctx,
                           float& x1, float& y1, float& x2, float& y2,
                           std::vector<Keypoint>& landmarks) const {
        if (ctx.pad_x == 0 && ctx.pad_y == 0) {
            float sx = static_cast<float>(ctx.original_width)  / input_width_;
            float sy = static_cast<float>(ctx.original_height) / input_height_;
            x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy;
            for (auto& kp : landmarks) { kp.x *= sx; kp.y *= sy; }
        } else {
            x1 = (x1 - ctx.pad_x) / ctx.scale;
            y1 = (y1 - ctx.pad_y) / ctx.scale;
            x2 = (x2 - ctx.pad_x) / ctx.scale;
            y2 = (y2 - ctx.pad_y) / ctx.scale;
            for (auto& kp : landmarks) {
                kp.x = (kp.x - ctx.pad_x) / ctx.scale;
                kp.y = (kp.y - ctx.pad_y) / ctx.scale;
            }
        }
    }

    void generatePriorBoxes() {
        int feature_map_strides[] = {8, 16, 32};
        int min_sizes[][2] = {{16, 32}, {64, 128}, {256, 512}};

        auto generate_stride_priors = [&](int stride, const int* ms, int fh, int fw) {
            for (int y = 0; y < fh; ++y) {
                for (int x = 0; x < fw; ++x) {
                    for (int k = 0; k < 2; ++k) {
                        float cx = (x + 0.5f) * stride / input_width_;
                        float cy = (y + 0.5f) * stride / input_height_;
                        float pw = static_cast<float>(ms[k]) / static_cast<float>(input_width_);
                        float ph = static_cast<float>(ms[k]) / static_cast<float>(input_height_);
                        priors_.push_back({cx, cy, pw, ph});
                    }
                }
            }
        };

        for (int s = 0; s < 3; ++s) {
            int stride = feature_map_strides[s];
            int fh = (input_height_ + stride - 1) / stride;
            int fw = (input_width_ + stride - 1) / stride;
            generate_stride_priors(stride, min_sizes[s], fh, fw);
        }
    }

    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    float variance_[2] = {0.1f, 0.2f};
    std::vector<std::array<float, 4>> priors_;
};

}  // namespace dxapp

#endif  // RETINAFACE_POSTPROCESSOR_HPP
