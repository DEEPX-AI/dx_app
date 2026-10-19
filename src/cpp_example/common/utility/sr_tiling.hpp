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
 * **Streaming.** runAndAssembleTiles() stitches each tile as soon as its output
 * is waited for, so only the in-flight window's outputs exist at once. Holding
 * every output until the frame is done costs ~0.5 GB at 1920x1080 (25,560
 * ESPCN x4 tiles).
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
#include <utility>
#include <vector>

#include "common/utility/colorspace.hpp"

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
 * @brief Run every planned tile through @p ie, pipelined via RunAsync, and hand
 *        each tile's outputs to `on_output(index, outputs)` as soon as they are
 *        waited for.
 *
 * Tiles complete in submission order. `outputs` is empty if that tile failed.
 * The tile buffer must outlive its job, so each clone is held until its Wait().
 * If RunAsync throws, every job already submitted is waited for (and passed to
 * @p on_output) before the exception leaves this function.
 *
 * @warning Only valid when NO callback is registered on @p ie. With a callback
 * installed the engine routes outputs there and Wait() returns nothing, so every
 * tile would come back empty.
 *
 * @tparam Engine dxrt::InferenceEngine, or anything with `int RunAsync(void*)`
 *         and `dxrt::TensorPtrs Wait(int)` (the unit tests' fake engines).
 * @tparam OnOutput callable as `on_output(size_t index, dxrt::TensorPtrs outputs)`.
 */
template <typename Engine, typename OnOutput>
inline void pipelineTiles(Engine& ie, const cv::Mat& lr_plane,
                          const std::vector<TilePlan>& plans,
                          int tile_h, int tile_w, int inflight, OnOutput on_output) {
    if (inflight < 1) inflight = 1;

    // (index, job id, tile buffer kept alive until Wait)
    std::deque<std::tuple<size_t, int, cv::Mat>> queue;

    auto drain_one = [&]() {
        auto item = queue.front();
        queue.pop_front();
        dxrt::TensorPtrs outputs;
        try {
            outputs = ie.Wait(std::get<1>(item));
        } catch (...) {
            outputs.clear();
        }
        on_output(std::get<0>(item), std::move(outputs));
    };

    try {
        for (size_t i = 0; i < plans.size(); ++i) {
            const TilePlan& p = plans[i];
            cv::Mat tile = lr_plane(cv::Rect(p.win_x, p.win_y, tile_w, tile_h)).clone();
            // The queue slot first: once RunAsync has submitted the job,
            // nothing that can throw stands between it and the queue, so
            // every submitted job is waited for below.
            queue.emplace_back(i, -1, tile);
            try {
                std::get<1>(queue.back()) = ie.RunAsync(std::get<2>(queue.back()).data);
            } catch (...) {
                queue.pop_back();  // never submitted
                throw;
            }
            if (static_cast<int>(queue.size()) >= inflight) drain_one();
        }
    } catch (...) {
        // A submit failed: the jobs already in flight still read their tile
        // buffers, so wait for every one before `queue` frees them.
        while (!queue.empty()) drain_one();
        throw;
    }
    while (!queue.empty()) drain_one();
}

/**
 * @brief pipelineTiles() collecting every tile's outputs into @p outputs, in
 *        tile order; an entry is empty if that tile failed.
 *
 * Holds all of a frame's outputs at once. The runners use runAndAssembleTiles().
 */
template <typename Engine>
inline void runTilesPipelined(Engine& ie, const cv::Mat& lr_plane,
                              const std::vector<TilePlan>& plans,
                              int tile_h, int tile_w,
                              std::vector<dxrt::TensorPtrs>& outputs,
                              int inflight = kDefaultInflight) {
    outputs.assign(plans.size(), dxrt::TensorPtrs{});
    pipelineTiles(ie, lr_plane, plans, tile_h, tile_w, inflight,
                  [&outputs](size_t i, dxrt::TensorPtrs tile_outputs) {
                      outputs[i] = std::move(tile_outputs);
                  });
}

/**
 * @brief Write tile @p p's valid region into @p sr_y (CV_8UC1, padded output size).
 *
 * Only the valid region is written, so with a halo of at least the
 * receptive-field radius the result carries no tile seams. False when the tile
 * has no output to stitch.
 */
inline bool assembleTile(const TilePlan& p, const dxrt::TensorPtrs& outputs,
                         int scale_y, int scale_x, int out_tile_w, cv::Mat& sr_y) {
    if (outputs.empty()) return false;
    const float* data = static_cast<const float*>(outputs[0]->data());
    if (data == nullptr) return false;

    const int sy = p.src_y * scale_y, sx = p.src_x * scale_x;
    const int vh = p.valid_h * scale_y, vw = p.valid_w * scale_x;
    const int dy = (p.win_y + p.src_y) * scale_y;
    const int dx = (p.win_x + p.src_x) * scale_x;

    for (int py = 0; py < vh; ++py) {
        const int oy = dy + py;
        if (oy >= sr_y.rows) break;
        uchar* dst = sr_y.ptr<uchar>(oy);
        const float* src = data + static_cast<size_t>(sy + py) * out_tile_w + sx;
        for (int px = 0; px < vw; ++px) {
            const int ox = dx + px;
            if (ox >= sr_y.cols) break;
            const float v = std::max(0.0f, std::min(1.0f, src[px]));
            dst[ox] = static_cast<uchar>(v * 255.0f + 0.5f);
        }
    }
    return true;
}

/**
 * @brief Run every planned tile through @p ie and stitch it into @p sr_y as soon
 *        as its output arrives, so only the in-flight window's outputs are alive.
 *
 * Same image as runTilesPipelined() + assembleTiles(). Returns the number of
 * tiles that contributed. Same engine requirement as pipelineTiles().
 */
template <typename Engine>
inline int runAndAssembleTiles(Engine& ie, const cv::Mat& lr_plane,
                               const std::vector<TilePlan>& plans,
                               int tile_h, int tile_w,
                               int padded_out_h, int padded_out_w,
                               int scale_y, int scale_x, int out_tile_w,
                               cv::Mat& sr_y, int inflight = kDefaultInflight) {
    sr_y = cv::Mat::zeros(padded_out_h, padded_out_w, CV_8UC1);
    int tiles_done = 0;
    pipelineTiles(ie, lr_plane, plans, tile_h, tile_w, inflight,
                  [&](size_t i, dxrt::TensorPtrs tile_outputs) {
                      if (assembleTile(plans[i], tile_outputs, scale_y, scale_x,
                                       out_tile_w, sr_y)) {
                          ++tiles_done;
                      }
                  });
    return tiles_done;
}

/**
 * @brief Stitch collected tile outputs into one SR luminance plane (CV_8UC1).
 *
 * Returns the number of tiles that contributed.
 */
inline int assembleTiles(const std::vector<TilePlan>& plans,
                         const std::vector<dxrt::TensorPtrs>& outputs,
                         int padded_out_h, int padded_out_w,
                         int scale_y, int scale_x,
                         int out_tile_w, cv::Mat& sr_y) {
    sr_y = cv::Mat::zeros(padded_out_h, padded_out_w, CV_8UC1);
    int tiles_done = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        if (assembleTile(plans[i], outputs[i], scale_y, scale_x, out_tile_w, sr_y)) {
            ++tiles_done;
        }
    }
    return tiles_done;
}

/** Compute output scale factors by probing once.
 *  Only the output *shape* matters, so a zero tile is enough.
 *  @return (scale_x, scale_y), each >= 1. */
inline std::pair<int, int> probeOutputScale(dxrt::InferenceEngine& ie, int tile_w, int tile_h) {
    cv::Mat probe_tile = cv::Mat::zeros(tile_h, tile_w, CV_8UC1);
    dxrt::TensorPtrs probe_out = ie.Run(probe_tile.data, nullptr, nullptr);
    int out_tile_h = tile_h, out_tile_w = tile_w;
    if (!probe_out.empty()) {
        auto shape = probe_out[0]->shape();
        if (shape.size() == 4)      { out_tile_h = static_cast<int>(shape[2]); out_tile_w = static_cast<int>(shape[3]); }
        else if (shape.size() == 3) { out_tile_h = static_cast<int>(shape[1]); out_tile_w = static_cast<int>(shape[2]); }
        else if (shape.size() == 2) { out_tile_h = static_cast<int>(shape[0]); out_tile_w = static_cast<int>(shape[1]); }
    }
    return {std::max(1, out_tile_w / tile_w), std::max(1, out_tile_h / tile_h)};
}

/// Replicate-pad `bgr` to padded_h x padded_w and take its limited-range Y
/// (ESPCN is trained on MATLAB rgb2ycbcr Y), exactly as the runner does.
inline void prepareLowRes(const cv::Mat& bgr, int padded_h, int padded_w,
                          cv::Mat& lr_bgr, cv::Mat& lr_gray) {
    cv::copyMakeBorder(bgr, lr_bgr, 0, padded_h - bgr.rows, 0, padded_w - bgr.cols,
                       cv::BORDER_REPLICATE);
    dxapp::colorspace::bgrToYLimited(lr_bgr, lr_gray);
}

/// The super-resolved luma `sr_y` plus `orig_bgr`'s chroma, bicubic-resized
/// to sr_y's size, back to BGR in the limited-range convention.
inline cv::Mat mergeSrLuma(const cv::Mat& orig_bgr, const cv::Mat& sr_y) {
    cv::Mat lr_ycrcb;
    dxapp::colorspace::bgrToYCrCbLimited(orig_bgr, lr_ycrcb);
    std::vector<cv::Mat> ch;
    cv::split(lr_ycrcb, ch);
    cv::Mat cr_up, cb_up;
    cv::resize(ch[1], cr_up, sr_y.size(), 0, 0, cv::INTER_CUBIC);
    cv::resize(ch[2], cb_up, sr_y.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat merged;
    cv::merge(std::vector<cv::Mat>{sr_y, cr_up, cb_up}, merged);
    cv::Mat bgr;
    dxapp::colorspace::ycrcbLimitedToBgr(merged, bgr);
    return bgr;
}

}  // namespace srtiling
}  // namespace dxapp

#endif  // DXAPP_SR_TILING_HPP
