/**
 * @file sr_tiling.hpp
 * @brief Tiled super-resolution: tile geometry and pipelined tile execution.
 *
 * Models compiled at a small fixed input (e.g. ESPCN at 17x17) can only cover a
 * larger image by tiling. Two things decide whether that looks right and runs
 * fast:
 *
 * **Overlap.** A CNN's output pixel needs its whole receptive field. Cut tiles
 * with no overlap and every pixel near a tile edge is reconstructed from the
 * model's internal zero-padding instead of the real neighbours, so a grid
 * appears at the tile period. Cutting with a `halo` of at least the
 * receptive-field radius and keeping only the valid centre removes the seams by
 * construction — no blending required. For ESPCN (conv 5x5 -> 3x3 -> 3x3) the
 * receptive field is 9x9, so the radius, and the correct halo, is 4.
 *
 * **Pipelining.** A 17x17 ESPCN forward is ~0.1 MFLOP, so a blocking Run() per
 * tile is almost entirely call overhead. Issuing RunAsync() over a bounded
 * in-flight window recovers ~4x throughput, which is what makes the extra tiles
 * from overlapping affordable.
 *
 * `halo=0` reduces this to plain non-overlapping tiling — the behaviour the
 * runners had before this header existed, useful as a regression harness.
 *
 * This is the C++ mirror of src/python_example/common/runner/sr_tiling.py —
 * keep the two in sync.
 */

#ifndef DXAPP_SR_TILING_HPP
#define DXAPP_SR_TILING_HPP

#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace dxapp {
namespace srtiling {

/// Receptive-field radius of ESPCN (conv 5x5 -> 3x3 -> 3x3 gives RF 9x9).
constexpr int kDefaultHalo = 4;

/// Highest halo accepted. 4 px of context is already everything an output pixel
/// can use, so a larger halo buys no accuracy while the stride (tile - 2 * halo)
/// collapses and the tile count explodes (17x17 tiles over a 275x150 frame:
/// 480 tiles at halo 4, 34,706 at halo 8).
constexpr int kMaxHalo = 4;

/// In-flight window for RunAsync. Throughput plateaus around 16-32.
constexpr int kDefaultInflight = 16;

struct TilePlan {
    int win_y;    ///< window origin in the padded LR plane
    int win_x;
    int src_y;    ///< offset of the valid region inside the window (LR px)
    int src_x;
    int valid_h;  ///< size of the valid region (LR px)
    int valid_w;
};

/// Halo in LR pixels, overridable by DXAPP_SR_TILE_HALO for testing. 0 selects
/// plain non-overlapping tiling.
inline int resolveHalo(int fallback = kDefaultHalo) {
    const char* raw = std::getenv("DXAPP_SR_TILE_HALO");
    if (raw == nullptr || *raw == '\0') return fallback;
    try {
        int value = std::stoi(std::string(raw));
        return value >= 0 ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

/// Largest halo that still leaves a positive stride for @p tile.
inline int maxHaloForTile(int tile) { return tile > 1 ? (tile - 1) / 2 : 0; }

/**
 * @brief Resolve the tile halo: CLI > config.json > DXAPP_SR_TILE_HALO > fallback.
 *
 * @p cli and @p cfg use -1 as the "not given" sentinel (C++14 — no std::optional),
 * so 0 stays a real value meaning "no overlap".
 *
 * An explicitly given value is strict: one that is negative, or too large for the
 * tile, leaves @p error non-empty so the caller can reject the user's input rather
 * than silently ignoring it. A malformed *environment* value keeps the lenient
 * historical behaviour and falls back to @p fallback.
 *
 * @param source out: where the value came from, for logging.
 * @param error  out: empty on success, else a user-facing message.
 */
inline int resolveHaloFrom(int cli, int cfg, int tile_h, int tile_w,
                           std::string& source, std::string& error,
                           int fallback = kDefaultHalo) {
    int halo = fallback;
    source = "default";
    if (cli < -1 || cfg < -1) {
        error = "SR tile halo must be >= 0";
        return fallback;
    }
    if (cli >= 0) {
        halo = cli;
        source = "--sr-tile-halo";
    } else if (cfg >= 0) {
        halo = cfg;
        source = "config.json";
    } else {
        const char* raw = std::getenv("DXAPP_SR_TILE_HALO");
        if (raw != nullptr && *raw != '\0') {
            try {
                int value = std::stoi(std::string(raw));
                if (value >= 0) {
                    halo = value;
                    source = "env DXAPP_SR_TILE_HALO";
                }
            } catch (...) {
                // keep the fallback
            }
        }
    }

    const int tile_limit = std::min(maxHaloForTile(tile_h), maxHaloForTile(tile_w));
    const int limit = std::min(kMaxHalo, tile_limit);
    if (halo > limit) {
        if (tile_limit < kMaxHalo) {
            error = "SR tile halo " + std::to_string(halo) + " is too large for a "
                  + std::to_string(tile_w) + "x" + std::to_string(tile_h)
                  + " tile (stride would be <= 0); use 0.." + std::to_string(limit);
        } else {
            error = "SR tile halo " + std::to_string(halo)
                  + " exceeds the maximum useful overlap; use 0.."
                  + std::to_string(kMaxHalo) + " (" + std::to_string(kMaxHalo)
                  + " is the receptive-field radius — a larger halo adds no accuracy"
                    " and multiplies the tile count)";
        }
        return fallback;
    }
    return halo;
}

/**
 * @brief Lay out windows along one axis.
 *
 * Fills @p windows with (win, src, valid) triples and returns the padded extent.
 * The valid regions are contiguous and together cover [0, padded).
 *
 * A window at the start of the plane, or ending exactly on its far edge, keeps
 * its border pixels: there is no neighbouring tile to supply that context, so
 * the model's own padding should act there — the same thing it sees when the
 * whole image goes through in one pass.
 */
inline int axisWindows(int orig, int tile, int halo,
                       std::vector<std::tuple<int, int, int>>& windows) {
    const int stride = tile - 2 * halo;
    if (stride <= 0) {
        throw std::invalid_argument("sr_tiling: halo too large for tile size");
    }

    const int steps = (orig <= tile) ? 0 : (orig - tile + stride - 1) / stride;
    const int padded = tile + steps * stride;

    windows.clear();
    windows.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i <= steps; ++i) {
        const int win = i * stride;
        const int src = (win == 0) ? 0 : halo;
        const int end = (win + tile == padded) ? tile : tile - halo;
        windows.emplace_back(win, src, end - src);
    }
    return padded;
}

/**
 * @brief Plan the tiling of an @p orig_h x @p orig_w LR plane.
 *
 * The caller must pad the LR plane to @p padded_h x @p padded_w before reading
 * windows from it.
 */
inline void planTiles(int orig_h, int orig_w, int tile_h, int tile_w, int halo,
                      int& padded_h, int& padded_w,
                      std::vector<TilePlan>& plans) {
    std::vector<std::tuple<int, int, int>> rows, cols;
    padded_h = axisWindows(orig_h, tile_h, halo, rows);
    padded_w = axisWindows(orig_w, tile_w, halo, cols);

    plans.clear();
    plans.reserve(rows.size() * cols.size());
    for (const auto& r : rows) {
        for (const auto& c : cols) {
            TilePlan p;
            p.win_y = std::get<0>(r); p.src_y = std::get<1>(r); p.valid_h = std::get<2>(r);
            p.win_x = std::get<0>(c); p.src_x = std::get<1>(c); p.valid_w = std::get<2>(c);
            plans.push_back(p);
        }
    }
}

/**
 * @brief Run every planned tile through @p ie, blocking on each.
 *
 * Use this when the engine has a callback registered (RegisterCallback): the
 * callback consumes the outputs, so RunAsync + Wait() would hand back empty
 * tensors. The async restoration runner is in exactly that situation.
 */
inline void runTilesBlocking(dxrt::InferenceEngine& ie, const cv::Mat& lr_plane,
                             const std::vector<TilePlan>& plans,
                             int tile_h, int tile_w,
                             std::vector<dxrt::TensorPtrs>& outputs) {
    outputs.assign(plans.size(), dxrt::TensorPtrs{});
    for (size_t i = 0; i < plans.size(); ++i) {
        const TilePlan& p = plans[i];
        cv::Mat tile = lr_plane(cv::Rect(p.win_x, p.win_y, tile_w, tile_h)).clone();
        try {
            outputs[i] = ie.Run(tile.data, nullptr, nullptr);
        } catch (...) {
            outputs[i] = dxrt::TensorPtrs{};
        }
    }
}

/**
 * @brief Run every planned tile through @p ie, pipelined via RunAsync.
 *
 * Outputs come back in tile order; an entry is empty if that tile failed. The
 * tile buffer must outlive its job, so each clone is held until its Wait().
 *
 * @warning Only valid when NO callback is registered on @p ie. With a callback
 * installed the engine routes outputs there and Wait() returns nothing, so every
 * tile would come back empty — use runTilesBlocking() instead.
 */
inline void runTilesPipelined(dxrt::InferenceEngine& ie, const cv::Mat& lr_plane,
                              const std::vector<TilePlan>& plans,
                              int tile_h, int tile_w,
                              std::vector<dxrt::TensorPtrs>& outputs,
                              int inflight = kDefaultInflight) {
    if (inflight < 1) inflight = 1;
    outputs.assign(plans.size(), dxrt::TensorPtrs{});

    // (index, job id, tile buffer kept alive until Wait)
    std::deque<std::tuple<size_t, int, cv::Mat>> queue;

    auto drain_one = [&]() {
        auto item = queue.front();
        queue.pop_front();
        try {
            outputs[std::get<0>(item)] = ie.Wait(std::get<1>(item));
        } catch (...) {
            outputs[std::get<0>(item)] = dxrt::TensorPtrs{};
        }
    };

    for (size_t i = 0; i < plans.size(); ++i) {
        const TilePlan& p = plans[i];
        cv::Mat tile = lr_plane(cv::Rect(p.win_x, p.win_y, tile_w, tile_h)).clone();
        const int job_id = ie.RunAsync(tile.data);
        queue.emplace_back(i, job_id, tile);
        if (static_cast<int>(queue.size()) >= inflight) drain_one();
    }
    while (!queue.empty()) drain_one();
}

/**
 * @brief Stitch tile outputs into one SR luminance plane (CV_8UC1).
 *
 * Only each tile's valid region is written, so with a halo of at least the
 * receptive-field radius the result carries no tile seams. Returns the number of
 * tiles that contributed.
 */
inline int assembleTiles(const std::vector<TilePlan>& plans,
                         const std::vector<dxrt::TensorPtrs>& outputs,
                         int padded_out_h, int padded_out_w,
                         int scale_y, int scale_x,
                         int out_tile_w, cv::Mat& sr_y) {
    sr_y = cv::Mat::zeros(padded_out_h, padded_out_w, CV_8UC1);
    int tiles_done = 0;

    for (size_t i = 0; i < plans.size(); ++i) {
        if (outputs[i].empty()) continue;
        const float* data = static_cast<const float*>(outputs[i][0]->data());
        if (data == nullptr) continue;

        const TilePlan& p = plans[i];
        const int sy = p.src_y * scale_y, sx = p.src_x * scale_x;
        const int vh = p.valid_h * scale_y, vw = p.valid_w * scale_x;
        const int dy = (p.win_y + p.src_y) * scale_y;
        const int dx = (p.win_x + p.src_x) * scale_x;

        for (int py = 0; py < vh; ++py) {
            const int oy = dy + py;
            if (oy >= padded_out_h) break;
            uchar* dst = sr_y.ptr<uchar>(oy);
            const float* src = data + static_cast<size_t>(sy + py) * out_tile_w + sx;
            for (int px = 0; px < vw; ++px) {
                const int ox = dx + px;
                if (ox >= padded_out_w) break;
                const float v = std::max(0.0f, std::min(1.0f, src[px]));
                dst[ox] = static_cast<uchar>(v * 255.0f + 0.5f);
            }
        }
        ++tiles_done;
    }
    return tiles_done;
}

}  // namespace srtiling
}  // namespace dxapp

#endif  // DXAPP_SR_TILING_HPP
