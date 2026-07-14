/**
 * @file superpoint_tracker.hpp
 * @brief Sparse optical flow tracker for SuperPoint keypoints.
 *
 * Ports the PointTracker class from the original Magic Leap SuperPoint demo
 * (DeTone & Malisiewicz, 2018) to C++.  Tracks keypoints across consecutive
 * frames using two-way nearest-neighbour descriptor matching and overlays
 * colour-coded track lines on the output image.
 */

#ifndef SUPERPOINT_TRACKER_HPP
#define SUPERPOINT_TRACKER_HPP

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace dxapp {

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
                               float max_pixel_dist = 100.f)
        : maxl_(max_length),
          nn_thresh_(nn_thresh),
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

        // --- Store descriptors for next frame ---
        last_desc_ = descs;
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

            const float score_norm = std::max(0.f, std::min(1.f, row[1]));
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
     * Matches go from last frame → current frame; only mutual bests below nn_thresh_.
     */
    std::vector<Match> nn_match_two_way(
            const std::vector<std::vector<float>>& descs) const {
        const int N1 = last_desc_count_;
        const int N2 = static_cast<int>(descs.size());
        if (N1 == 0 || N2 == 0) return {};

        // Distance matrix: D[i][j] = L2(last_desc_[i], descs[j])
        // Using unit-normalised: L2 = sqrt(2 - 2*dot)
        std::vector<std::vector<float>> dmat(N1, std::vector<float>(N2));
        for (int i = 0; i < N1; ++i) {
            const auto& d1 = last_desc_[i];
            for (int j = 0; j < N2; ++j) {
                float dot = 0.f;
                const auto& d2 = descs[j];
                for (int k = 0; k < kDescDim; ++k) dot += d1[k] * d2[k];
                dot = std::max(-1.f, std::min(1.f, dot));
                dmat[i][j] = std::sqrt(2.f - 2.f * dot);
            }
        }

        // NN from desc1 → desc2
        std::vector<int> nn12(N1);
        std::vector<float> score12(N1);
        for (int i = 0; i < N1; ++i) {
            int best = 0;
            for (int j = 1; j < N2; ++j)
                if (dmat[i][j] < dmat[i][best]) best = j;
            nn12[i] = best;
            score12[i] = dmat[i][best];
        }

        // NN from desc2 → desc1
        std::vector<int> nn21(N2);
        for (int j = 0; j < N2; ++j) {
            int best = 0;
            for (int i = 1; i < N1; ++i)
                if (dmat[i][j] < dmat[best][j]) best = i;
            nn21[j] = best;
        }

        // Mutual check + threshold
        std::vector<Match> matches;
        for (int i = 0; i < N1; ++i) {
            if (score12[i] < nn_thresh_ && nn21[nn12[i]] == i)
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
    float max_pixel_dist_sq_;
    int track_count_;
    float max_score_;
    int track_cols_;

    // all_pts_[frame_idx][kp_idx] = (x, y)
    std::vector<std::vector<std::pair<float, float>>> all_pts_;

    // Descriptor history for last frame: N x kDescDim
    std::vector<std::vector<float>> last_desc_;
    int last_desc_count_;

    // Track rows: each row = [id, score, pt_id_0, …, pt_id_{L-1}]
    std::vector<std::vector<float>> tracks_;

    mutable std::mutex mtx_;  // guards all mutable state for thread-safety
};

}  // namespace dxapp

#endif  // SUPERPOINT_TRACKER_HPP
