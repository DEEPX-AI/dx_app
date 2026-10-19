/**
 * @file superpoint_tracker.hpp
 * @brief Sparse optical flow tracker for SuperPoint keypoints.
 *
 * Ports the PointTracker class from the original Magic Leap SuperPoint demo
 * (DeTone & Malisiewicz, 2018) to C++.  Tracks keypoints across consecutive
 * frames using two-way nearest-neighbour descriptor matching and overlays
 * colour-coded track lines on the output image.
 */

#ifndef SUPERPOINT_480X640_SUPERPOINT_TRACKER_HPP
#define SUPERPOINT_480X640_SUPERPOINT_TRACKER_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace dxapp {
namespace v_superpoint_480x640 {

// ---------------------------------------------------------------------------
// Jet colormap — 10 RGB entries, same as the original Python demo.
// Returns BGR scalar for OpenCV drawing.
// ---------------------------------------------------------------------------
static cv::Scalar jet_bgr(float val) {
    static const float R[10] = {0.f, 0.f, 0.f, 0.f,
                                 0.300f, 0.667f, 1.f, 1.f, 1.f, 0.5f};
    static const float G[10] = {0.f, 0.f, 0.378f, 0.833f,
                                 1.f, 1.f, 0.901f, 0.480f, 0.073f, 0.f};
    static const float B[10] = {0.5f, 1.f, 1.f, 1.f,
                                 0.667f, 0.300f, 0.f, 0.f, 0.f, 0.f};
    const int idx = std::max(0, std::min(9, static_cast<int>(val * 10.f)));
    return cv::Scalar(B[idx] * 255, G[idx] * 255, R[idx] * 255);
}

// ---------------------------------------------------------------------------
// SuperPointTracker
// ---------------------------------------------------------------------------
/**
 * @brief Manages sparse keypoint tracks across frames via descriptor matching.
 *
 * Tracks matrix row layout: [track_id, avg_score, pt_id_0, …, pt_id_{L-1}]
 * where L = max_length and pt_id values are global keypoint indices (-1 = unobserved).
 */
class SuperPointTracker {
public:
    static constexpr int kDescDim = 256;

    explicit SuperPointTracker(int max_length = 5, float nn_thresh = 0.7f,
                               float max_pixel_dist = 100.f,
                               float ratio_thresh = 0.75f)
        : maxl_(max_length),
          nn_thresh_(nn_thresh),
          ratio_thresh_(ratio_thresh),
          max_pixel_dist_sq_(max_pixel_dist * max_pixel_dist),
          track_count_(0),
          max_score_(9999.f),
          last_desc_count_(0) {
        all_pts_.resize(max_length);          // each entry: vector of (x,y)
        track_cols_ = max_length + 2;         // [id, score, pt_0, …, pt_{L-1}]
    }

    /**
     * @brief Add a new frame's keypoints and descriptors; update tracks.
     *
     * @param xs        x-coordinates (original image space), length N
     * @param ys        y-coordinates (original image space), length N
     * @param descs     descriptor matrix; descs[i] has kDescDim floats, length N
     */
    void update(const std::vector<float>& xs,
                const std::vector<float>& ys,
                const std::vector<std::vector<float>>& descs) {
        std::lock_guard<std::mutex> lock(mtx_);
        const int N = static_cast<int>(xs.size());

        // --- Roll frame history ---
        const int remove_size = static_cast<int>(all_pts_.front().size());
        all_pts_.erase(all_pts_.begin());
        std::vector<std::pair<float, float>> cur(N);
        for (int i = 0; i < N; ++i) cur[i] = {xs[i], ys[i]};
        all_pts_.push_back(std::move(cur));

        // --- Drop oldest column from tracks, adjust indices ---
        if (!tracks_.empty()) {
            // Delete column index 2 (the oldest pt_id column)
            const int old_rows = static_cast<int>(tracks_.size());
            for (int r = 0; r < old_rows; ++r) {
                tracks_[r].erase(tracks_[r].begin() + 2);
                // Adjust point IDs for the removed frame
                for (int c = 2; c < static_cast<int>(tracks_[r].size()); ++c) {
                    if (tracks_[r][c] >= 0.f)
                        tracks_[r][c] -= static_cast<float>(remove_size);
                    if (tracks_[r][c] < -1.f)
                        tracks_[r][c] = -1.f;
                }
                // Append placeholder for new frame
                tracks_[r].push_back(-1.f);
            }
        }

        // --- Compute frame offsets ---
        std::vector<int> offsets = compute_offsets();

        // --- Two-way NN matching ---
        std::vector<bool> matched(N, false);
        if (last_desc_count_ > 0 && N > 0) {
            // matches: each entry = {idx1, idx2, score}
            auto matches = nn_match_two_way(descs);
            const auto& prev_frame = all_pts_[static_cast<int>(all_pts_.size()) - 2];
            for (const auto& m : matches) {
                // Skip if pixel distance between matched points is too large
                if (!prev_frame.empty() && max_pixel_dist_sq_ > 0.f) {
                    const float dx = xs[m.idx2] - prev_frame[m.idx1].first;
                    const float dy = ys[m.idx2] - prev_frame[m.idx1].second;
                    if (dx * dx + dy * dy > max_pixel_dist_sq_) continue;
                }
                const int id1 = m.idx1 + offsets[static_cast<int>(offsets.size()) - 2];
                const int id2 = m.idx2 + offsets[static_cast<int>(offsets.size()) - 1];
                // Find track whose last pt_id matches id1
                for (auto& row : tracks_) {
                    if (static_cast<int>(row[row.size() - 2]) == id1) {
                        matched[m.idx2] = true;
                        row.back() = static_cast<float>(id2);
                        if (row[1] == max_score_) {
                            row[1] = m.score;
                        } else {
                            int obs = 0;
                            for (int c = 2; c < static_cast<int>(row.size()); ++c)
                                if (row[c] >= 0.f) ++obs;
                            const float track_len = static_cast<float>(obs - 1);
                            const float frac = track_len > 0.f ? 1.f / track_len : 1.f;
                            row[1] = (1.f - frac) * row[1] + frac * m.score;
                        }
                        break;
                    }
                }
            }
        }

        // --- Add unmatched keypoints as new tracks ---
        const int off_last = offsets.back();
        for (int i = 0; i < N; ++i) {
            if (matched[i]) continue;
            std::vector<float> row(track_cols_, -1.f);
            row[0] = static_cast<float>(track_count_++);
            row[1] = max_score_;
            row.back() = static_cast<float>(i + off_last);
            tracks_.push_back(std::move(row));
        }

        // --- Prune tracks with no valid observations ---
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(), [](const std::vector<float>& row) {
                for (int c = 2; c < static_cast<int>(row.size()); ++c)
                    if (row[c] >= 0.f) return false;
                return true;
            }),
            tracks_.end());

        // --- Store descriptors for next frame (flat, for nn_match_two_way) ---
        last_desc_flat_.resize(static_cast<size_t>(N) * kDescDim);
        for (int i = 0; i < N; ++i)
            std::copy(descs[i].begin(), descs[i].begin() + kDescDim,
                      last_desc_flat_.begin() + static_cast<size_t>(i) * kDescDim);
        last_desc_count_ = N;
    }

    /**
     * @brief Draw coloured track lines on @p out (in-place BGR).
     * @param min_length Only draw tracks with >= min_length consecutive observations.
     */
    void drawTracks(cv::Mat& out, int min_length = 2) const {
        std::lock_guard<std::mutex> lock(mtx_);
        const int n_frames = static_cast<int>(all_pts_.size());
        const std::vector<int> offsets = compute_offsets();

        for (const auto& row : tracks_) {
            // Count observations
            int obs = 0;
            for (int c = 2; c < static_cast<int>(row.size()); ++c)
                if (row[c] >= 0.f) ++obs;
            if (obs < min_length) continue;
            if (row.back() < 0.f) continue;  // no observation in latest frame

            // Match scores live in [0, nn_thresh) by construction, so normalise
            // by nn_thresh before the colormap lookup -- exactly what the Python
            // visualizer does. Without it the jet scale was only ever driven to
            // index 6, compressing the colour range and putting the two
            // implementations one colour step apart on identical input.
            const float score_norm = std::max(
                0.f, std::min(1.f, row[1] / std::max(nn_thresh_, 1e-6f)));
            const cv::Scalar color = jet_bgr(score_norm);

            for (int i = 0; i < n_frames - 1; ++i) {
                if (row[i + 2] < 0.f || row[i + 3] < 0.f) continue;
                const int idx1 = static_cast<int>(row[i + 2]) - offsets[i];
                const int idx2 = static_cast<int>(row[i + 3]) - offsets[i + 1];
                if (idx1 < 0 || idx1 >= static_cast<int>(all_pts_[i].size())) continue;
                if (idx2 < 0 || idx2 >= static_cast<int>(all_pts_[i + 1].size())) continue;

                const auto& p1 = all_pts_[i][idx1];
                const auto& p2 = all_pts_[i + 1][idx2];
                const cv::Point cp1(static_cast<int>(std::round(p1.first)),
                                    static_cast<int>(std::round(p1.second)));
                const cv::Point cp2(static_cast<int>(std::round(p2.first)),
                                    static_cast<int>(std::round(p2.second)));
                cv::line(out, cp1, cp2, color, 1, cv::LINE_AA);
                if (i == n_frames - 2) {
                    cv::circle(out, cp2, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                }
            }
        }
    }

private:
    struct Match { int idx1; int idx2; float score; };

    /**
     * @brief Two-way nearest-neighbour matching between last_desc_ and descs.
     *
     * A match must be (a) closer than nn_thresh_, (b) mutually nearest, and
     * (c) clearly better than the runner-up -- Lowe's ratio test,
     * d_best < ratio_thresh_ * d_second.
     *
     * (c) is what rejects repetitive structure. On a glass office facade the
     * best and runner-up descriptors are near-tied, so the match lands on the
     * wrong window: measured on dashcam footage, matches that jumped >25 px had
     * a median ratio of 0.807 against 0.254 for ordinary matches, while their
     * absolute distance (0.584) still cleared nn_thresh = 0.70. A ratio of 0.75
     * keeps 96.9% of matches and removes 68% of those jumps.
     */
    std::vector<Match> nn_match_two_way(
            const std::vector<std::vector<float>>& descs) const {
        const int N1 = last_desc_count_;
        const int N2 = static_cast<int>(descs.size());
        if (N1 == 0 || N2 == 0) return {};

        // Distance matrix: D[i*N2 + j] = L2(last_desc_[i], descs[j])
        // Using unit-normalised: L2 = sqrt(2 - 2*dot)
        //
        // descs is a vector<vector<float>>, so each descriptor is its own heap
        // block: an i-j-k loop over it re-reads a different pointer for every j
        // and cannot vectorise. Transposing the current frame into one flat
        // 256 x N2 buffer and accumulating k-i-j makes the innermost loop a
        // contiguous AXPY over j, which the compiler does vectorise -- measured
        // 56 ms -> 7 ms per frame at 500x500x256. Each dot product still sums
        // k = 0..255 in the same order, so the results are unchanged.
        std::vector<float> bt(static_cast<size_t>(kDescDim) * N2);
        for (int j = 0; j < N2; ++j) {
            const auto& d2 = descs[j];
            for (int k = 0; k < kDescDim; ++k) bt[static_cast<size_t>(k) * N2 + j] = d2[k];
        }

        std::vector<float> dmat(static_cast<size_t>(N1) * N2, 0.f);
        for (int i = 0; i < N1; ++i) {
            const float* d1 = last_desc_flat_.data() + static_cast<size_t>(i) * kDescDim;
            float* row = dmat.data() + static_cast<size_t>(i) * N2;
            for (int k = 0; k < kDescDim; ++k) {
                const float a = d1[k];
                const float* b = bt.data() + static_cast<size_t>(k) * N2;
                for (int j = 0; j < N2; ++j) row[j] += a * b[j];
            }
            // Clamping the dot product to <= 1 already makes the radicand
            // 2 - 2*dot >= 0 (exactly 0 at dot == 1, since 2 - 2*1 is exact in
            // binary floating point), so sqrt never sees a negative argument
            // and needs no extra max(0, ...) guard.
            for (int j = 0; j < N2; ++j)
                row[j] = std::sqrt(2.f - 2.f * std::max(-1.f, std::min(1.f, row[j])));
        }

        // NN from desc1 → desc2, tracking the runner-up for the ratio test.
        // With a single candidate (N2 == 1) the j-loop does not run, so
        // d_second stays +inf and the ratio test below passes unconditionally --
        // there is nothing to be ambiguous against.
        std::vector<int> nn12(N1);
        std::vector<float> score12(N1);
        std::vector<float> second12(N1);
        const float kInf = std::numeric_limits<float>::infinity();
        for (int i = 0; i < N1; ++i) {
            const float* row = dmat.data() + static_cast<size_t>(i) * N2;
            int best = 0;
            float d_best = row[0];
            float d_second = kInf;
            for (int j = 1; j < N2; ++j) {
                if (row[j] < d_best) { d_second = d_best; d_best = row[j]; best = j; }
                else if (row[j] < d_second) { d_second = row[j]; }
            }
            nn12[i] = best;
            score12[i] = d_best;
            second12[i] = d_second;
        }

        // NN from desc2 → desc1
        std::vector<int> nn21(N2);
        for (int j = 0; j < N2; ++j) {
            int best = 0;
            for (int i = 1; i < N1; ++i)
                if (dmat[static_cast<size_t>(i) * N2 + j] <
                    dmat[static_cast<size_t>(best) * N2 + j]) best = i;
            nn21[j] = best;
        }

        // Mutual check + threshold
        std::vector<Match> matches;
        for (int i = 0; i < N1; ++i) {
            if (score12[i] >= nn_thresh_) continue;
            if (nn21[nn12[i]] != i) continue;
            // The > 0 guard is what disables the ratio test, and it has to
            // short-circuit: with ratio_thresh_ == 0 the product is 0 for a
            // finite runner-up and NaN for the +inf one, and "score < 0" /
            // "score < NaN" are both false -- without the guard, disabling the
            // test would reject every match instead of accepting them all.
            if (ratio_thresh_ > 0.f && !(score12[i] < ratio_thresh_ * second12[i])) continue;
            matches.push_back({i, nn12[i], score12[i]});
        }
        return matches;
    }

    std::vector<int> compute_offsets() const {
        const int n = static_cast<int>(all_pts_.size());
        std::vector<int> off(n, 0);
        for (int i = 1; i < n; ++i)
            off[i] = off[i - 1] + static_cast<int>(all_pts_[i - 1].size());
        return off;
    }

    int maxl_;
    float nn_thresh_;
    float ratio_thresh_;   // Lowe ratio test; <= 0 disables it
    float max_pixel_dist_sq_;
    int track_count_;
    float max_score_;
    int track_cols_;

    // all_pts_[frame_idx][kp_idx] = (x, y)
    std::vector<std::vector<std::pair<float, float>>> all_pts_;

    // Descriptor history for last frame, flattened row-major: N x kDescDim
    std::vector<float> last_desc_flat_;
    int last_desc_count_;

    // Track rows: each row = [id, score, pt_id_0, …, pt_id_{L-1}]
    std::vector<std::vector<float>> tracks_;

    mutable std::mutex mtx_;  // guards all mutable state for thread-safety
};

}  // namespace v_superpoint_480x640
}  // namespace dxapp

#endif  // SUPERPOINT_480X640_SUPERPOINT_TRACKER_HPP
