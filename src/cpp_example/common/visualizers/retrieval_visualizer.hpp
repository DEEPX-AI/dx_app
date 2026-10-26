/**
 * @file retrieval_visualizer.hpp
 * @brief Query beside its top-k gallery hits -- image retrieval, VPR, person ReID.
 *
 * The C++ counterpart of common/visualizers/retrieval_visualizer.py, laid out with the
 * same constants so the two trees render the same panel: query on the left in blue,
 * rank 1 framed green, each tile captioned with its cosine similarity, and a footer
 * naming the gallery that answered (so a stale gallery is visible in the picture).
 *
 * Unlike EmbeddingVisualizer this draws on the FIRST frame: the comparison set is the
 * committed gallery, not a previous frame. So these tasks produce an output image from
 * a single --image run, which the embedding comparison visualizer could not.
 */

#ifndef RETRIEVAL_VISUALIZER_HPP
#define RETRIEVAL_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include "common/utility/repo_path.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace dxapp {

namespace retrieval_vis_detail {

/// Resolve a repo-relative gallery path: as given, then under the repository
/// (RuntimeRoot(), as resolveGalleryFile opens the gallery itself), then from
/// successive parents of the working directory. Plain ifstream probing rather than
/// <filesystem>, because this tree still builds as C++14 on some targets.
inline std::string resolvePath(const std::string& path) {
    if (std::ifstream(path).good()) return path;
    const std::string rooted = ResolveRepoRelative(path);
    if (!rooted.empty() && std::ifstream(rooted).good()) return rooted;
    std::string prefix = "../";
    for (int depth = 0; depth < 8; ++depth) {
        const std::string candidate = prefix + path;
        if (std::ifstream(candidate).good()) return candidate;
        prefix += "../";
    }
    return path;
}

inline std::string baseName(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

inline std::string fmt(const char* spec, double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), spec, value);
    return std::string(buf);
}

}  // namespace retrieval_vis_detail

class RetrievalVisualizer : public IVisualizer<EmbeddingResult> {
public:
    explicit RetrievalVisualizer(std::string title = "Retrieval",
                                 int max_display = 3, int tile = 260)
        : title_(std::move(title)), max_display_(max_display), tile_(tile) {}

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<EmbeddingResult>& results,
                 const PreprocessContext& ctx) override {
        (void)ctx;
        if (results.empty()) return frame.clone();
        const auto& r = results[0];

        const int shown = std::min<int>(max_display_,
                                        static_cast<int>(r.matches.size()));
        const int box = tile_, pad = 14, head = 40, cap = 46, foot = 30;
        const int cols = 1 + std::max(1, shown);
        const int W = pad + cols * (box + pad);
        const int H = head + box + cap + foot;

        cv::Mat canvas(H, W, CV_8UC3, kBg());
        putText(canvas, title_, cv::Point(pad, 27), 0.72, kFg(), 2);

        // ---- query tile
        int x = pad;
        cv::Mat q = fit(frame, box);
        frameRect(q, kQuery(), 3);
        q.copyTo(canvas(cv::Rect(x, head, box, box)));
        putText(canvas, "QUERY", cv::Point(x + 4, head + box + 22), 0.58, kQuery(), 2);
        putText(canvas, std::to_string(r.dimension) + "-d descriptor",
                cv::Point(x + 4, head + box + 40), 0.46, kDim(), 1);

        if (shown == 0) {
            // No gallery: say so, and say how to build one, instead of an empty strip.
            x = pad + box + pad;
            cv::Mat tile(box, box, CV_8UC3, kBg());
            putText(tile, "no gallery", cv::Point(12, box / 2 - 10), 0.7, kDim(), 2);
            const char* lines[] = {"build one with:", "scripts/build_gallery_",
                                   "database.py"};
            for (int i = 0; i < 3; ++i)
                putText(tile, lines[i], cv::Point(12, box / 2 + 22 + i * 22), 0.46,
                        kDim(), 1);
            frameRect(tile, kDim(), 1);
            tile.copyTo(canvas(cv::Rect(x, head, box, box)));
        } else {
            for (int i = 0; i < shown; ++i) {
                const auto& m = r.matches[static_cast<std::size_t>(i)];
                x = pad + (i + 1) * (box + pad);
                const cv::Scalar colour = (i == 0) ? kAccent() : kFg();
                cv::Mat tile = load(m.path, box);
                frameRect(tile, colour, i == 0 ? 3 : 1);
                tile.copyTo(canvas(cv::Rect(x, head, box, box)));
                putText(canvas,
                        "#" + std::to_string(m.rank) + "  cos "
                            + retrieval_vis_detail::fmt("%.4f", m.score),
                        cv::Point(x + 4, head + box + 22), 0.56, colour, 2);
                std::string sub = m.label.empty()
                    ? retrieval_vis_detail::baseName(m.path) : m.label;
                if (sub.size() > 30) sub = sub.substr(0, 30);
                putText(canvas, sub, cv::Point(x + 4, head + box + 40), 0.46, kDim(), 1);
            }
        }

        const std::string gallery = r.gallery_name.empty() ? "-" : r.gallery_name;
        putText(canvas,
                "gallery: " + gallery + "  (" + std::to_string(r.gallery_size)
                    + " images)",
                cv::Point(pad, H - 10), 0.46, kDim(), 1);
        return canvas;
    }

    void setParameters(int line_thickness = 2, double font_scale = 0.5,
                       float alpha = 0.6f) override {
        (void)line_thickness; (void)font_scale; (void)alpha;
    }

private:
    // Function-local statics, not static data members: this header is included by one
    // translation unit per executable today, but an out-of-line definition in a header
    // is a duplicate symbol the moment two TUs in one binary include it, and C++14 has
    // no inline variables.
    static const cv::Scalar& kBg()     { static const cv::Scalar v(24, 24, 28);    return v; }
    static const cv::Scalar& kFg()     { static const cv::Scalar v(235, 235, 235); return v; }
    static const cv::Scalar& kDim()    { static const cv::Scalar v(150, 150, 155); return v; }
    static const cv::Scalar& kAccent() { static const cv::Scalar v(80, 200, 120);  return v; }
    static const cv::Scalar& kQuery()  { static const cv::Scalar v(90, 170, 245);  return v; }

    static void putText(cv::Mat& img, const std::string& text, cv::Point at,
                        double scale, const cv::Scalar& colour, int thickness) {
        cv::putText(img, text, at, cv::FONT_HERSHEY_SIMPLEX, scale, colour, thickness,
                    cv::LINE_AA);
    }

    /// Letterbox into a square tile, so aspect ratio is never misrepresented.
    cv::Mat fit(const cv::Mat& img, int box) const {
        cv::Mat canvas(box, box, CV_8UC3, kBg());
        if (img.empty()) return canvas;
        const double s = std::min(static_cast<double>(box) / img.cols,
                                  static_cast<double>(box) / img.rows);
        const int nw = std::max(1, static_cast<int>(img.cols * s + 0.5));
        const int nh = std::max(1, static_cast<int>(img.rows * s + 0.5));
        cv::Mat fittedTile;
        cv::resize(img, fittedTile, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
        fittedTile.copyTo(canvas(cv::Rect((box - nw) / 2, (box - nh) / 2, nw, nh)));
        return canvas;
    }

    /// A gallery tile, or a labelled placeholder: the gallery stores paths, not pixels,
    /// so a moved sample directory must show as a visible gap rather than a crash.
    cv::Mat load(const std::string& path, int box) const {
        cv::Mat img = cv::imread(retrieval_vis_detail::resolvePath(path),
                                 cv::IMREAD_COLOR);
        if (img.empty()) {
            cv::Mat tile(box, box, CV_8UC3, kBg());
            putText(tile, "missing", cv::Point(10, box / 2), 0.6, kDim(), 1);
            std::string name = retrieval_vis_detail::baseName(path);
            if (name.size() > 26) name = name.substr(0, 26);
            putText(tile, name, cv::Point(10, box / 2 + 24), 0.45, kDim(), 1);
            return tile;
        }
        return fit(img, box);
    }

    static void frameRect(cv::Mat& tile, const cv::Scalar& colour, int width) {
        cv::rectangle(tile, cv::Point(0, 0), cv::Point(tile.cols - 1, tile.rows - 1),
                      colour, width);
    }

    std::string title_;
    int max_display_;
    int tile_;
};

}  // namespace dxapp

#endif  // RETRIEVAL_VISUALIZER_HPP
