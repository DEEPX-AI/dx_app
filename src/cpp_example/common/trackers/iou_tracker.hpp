/**
 * @file iou_tracker.hpp
 * @brief Lightweight IoU multi-object tracker (SORT without Kalman)
 *
 * Assigns a stable track_id to per-frame bounding boxes so that instance
 * colors stay consistent across frames (same object -> same color) in the
 * instance segmentation visualizer. Association is greedy IoU matching between
 * the previous frame's tracks and the current frame's boxes. A track that goes
 * unmatched survives for max_age frames before being dropped, so a briefly
 * occluded object keeps its id/color when it reappears.
 *
 * Header-only, C++14, no dependencies beyond the STL.
 */

#ifndef DXAPP_IOU_TRACKER_HPP
#define DXAPP_IOU_TRACKER_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

namespace dxapp {

class IouTracker {
public:
    /**
     * @param iou_threshold minimum IoU for a box to match an existing track.
     * @param max_age consecutive unmatched frames a track survives before drop.
     */
    explicit IouTracker(float iou_threshold = 0.3f, int max_age = 30)
        : iou_threshold_(iou_threshold), max_age_(max_age) {}

    /** Clear all track state (e.g. between independent image sequences). */
    void reset() {
        tracks_.clear();
        next_id_ = 0;
    }

    /**
     * @brief Associate current-frame boxes with existing tracks.
     * @param boxes each box is [x1, y1, x2, y2] (size >= 4).
     * @return track_id per input box, order preserved. Unmatched boxes get a
     *         freshly allocated id.
     */
    std::vector<int> update(const std::vector<std::vector<float>>& boxes) {
        const std::size_t n = boxes.size();
        std::vector<int> track_ids(n, -1);
        if (n == 0) {
            ageAndPrune(std::vector<bool>(tracks_.size(), false));
            return track_ids;
        }

        // Collect candidate matches (iou, box_idx, track_idx) above threshold.
        struct Cand { float iou; std::size_t bi; std::size_t ti; };
        std::vector<Cand> candidates;
        for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
            for (std::size_t bi = 0; bi < n; ++bi) {
                float v = iou(tracks_[ti].box, boxes[bi]);
                if (v >= iou_threshold_) {
                    candidates.push_back(Cand{v, bi, ti});
                }
            }
        }

        // Greedy: highest IoU first, one-to-one.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Cand& a, const Cand& b) { return a.iou > b.iou; });

        std::vector<bool> box_matched(n, false);
        std::vector<bool> track_matched(tracks_.size(), false);
        for (const auto& c : candidates) {
            if (box_matched[c.bi] || track_matched[c.ti]) continue;
            box_matched[c.bi] = true;
            track_matched[c.ti] = true;
            Track& t = tracks_[c.ti];
            t.box = boxes[c.bi];
            t.time_since_update = 0;
            ++t.hits;
            track_ids[c.bi] = t.id;
        }

        // Unmatched boxes -> new tracks.
        for (std::size_t bi = 0; bi < n; ++bi) {
            if (box_matched[bi]) continue;
            Track t;
            t.id = next_id_++;
            t.box = boxes[bi];
            t.time_since_update = 0;
            t.hits = 1;
            tracks_.push_back(t);
            track_matched.push_back(true);  // keep aligned with tracks_
            track_ids[bi] = t.id;
        }

        ageAndPrune(track_matched);
        return track_ids;
    }

private:
    struct Track {
        int id{0};
        std::vector<float> box;
        int time_since_update{0};
        int hits{0};
    };

    static float iou(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() < 4 || b.size() < 4) return 0.0f;
        const float ix1 = std::max(a[0], b[0]);
        const float iy1 = std::max(a[1], b[1]);
        const float ix2 = std::min(a[2], b[2]);
        const float iy2 = std::min(a[3], b[3]);
        const float iw = ix2 - ix1;
        const float ih = iy2 - iy1;
        if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
        const float inter = iw * ih;
        const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
        const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
        const float uni = area_a + area_b - inter;
        return uni > 0.0f ? inter / uni : 0.0f;
    }

    void ageAndPrune(const std::vector<bool>& track_matched) {
        std::vector<Track> surviving;
        surviving.reserve(tracks_.size());
        for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
            Track t = tracks_[ti];
            if (ti >= track_matched.size() || !track_matched[ti]) {
                ++t.time_since_update;
            }
            if (t.time_since_update <= max_age_) {
                surviving.push_back(std::move(t));
            }
        }
        tracks_ = std::move(surviving);
    }

    float iou_threshold_;
    int max_age_;
    std::vector<Track> tracks_;
    int next_id_{0};
};

}  // namespace dxapp

#endif  // DXAPP_IOU_TRACKER_HPP
