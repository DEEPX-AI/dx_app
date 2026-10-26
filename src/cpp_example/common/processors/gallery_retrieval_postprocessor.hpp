/**
 * @file gallery_retrieval_postprocessor.hpp
 * @brief Rank a descriptor against a prebuilt gallery: retrieval, VPR and person ReID.
 *
 * An embedding model cannot answer "which place is this?" on its own -- it returns a
 * descriptor, and the answer needs a gallery of known descriptors to rank against. The
 * zoo's Image Retrieval, Visual Place Recognition and Person ReID categories are the
 * same operation over different galleries, so they share this class.
 *
 * The gallery is built ONCE, on the NPU, by scripts/build_gallery_database.py. That
 * script writes the same matrix twice from one in-memory array: a .npz for the Python
 * tree and the flat .bin read here. One call site, so the two cannot disagree.
 *
 * Format (little-endian, produced by write_gallery_bin):
 *   char[8]        "DXGAL1\0\0"   magic + format version
 *   uint32         n              entries
 *   uint32         dim            descriptor width
 *   float32[n*dim] embeddings     row-major, L2-normalised
 *   uint32 + bytes model          producing .dxnn stem
 *   uint32 + bytes paths          n lines, '\n'-joined, repo-relative
 *   uint32 + bytes labels         n lines, '\n'-joined (may be blank)
 *
 * Two refusals, because a mismatched gallery is invisible otherwise -- it does not
 * error, it ranks wrongly and returns confident numbers:
 *   dim    a gallery of another width cannot be multiplied at all.
 *   model  a gallery built by a DIFFERENT encoder is a different embedding space. The
 *          product still computes. eigenplaces-resnet18, pp-shituv2 and
 *          repvgg-a0-reid all emit 512-d, so shape alone catches nothing.
 */

#ifndef GALLERY_RETRIEVAL_POSTPROCESSOR_HPP
#define GALLERY_RETRIEVAL_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/utility/repo_path.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <filesystem>
namespace gallery_fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace gallery_fs = std::experimental::filesystem;
#endif

namespace dxapp {

/// One gallery, as loaded from a .bin. Kept separate from the postprocessor so the
/// parity test and the builder can exercise the reader on its own.
struct RetrievalGallery {
    std::vector<float> vectors;        // n * dim, row-major, unit norm
    int count{0};
    int dim{0};
    std::string model;                 // the .dxnn stem that produced it
    std::string name;                  // file name, for the footer
    std::vector<std::string> paths;
    std::vector<std::string> labels;
    bool loaded{false};
    std::string error;                 // why it is not loaded

    bool empty() const { return !loaded || count == 0; }
};

namespace gallery_detail {

inline std::vector<std::string> splitLines(const std::string& blob) {
    std::vector<std::string> out;
    if (blob.empty()) return out;
    std::size_t start = 0;
    while (true) {
        std::size_t nl = blob.find('\n', start);
        if (nl == std::string::npos) { out.push_back(blob.substr(start)); break; }
        out.push_back(blob.substr(start, nl - start));
        start = nl + 1;
    }
    return out;
}

inline bool readU32(std::ifstream& fh, std::uint32_t& value) {
    unsigned char raw[4];
    if (!fh.read(reinterpret_cast<char*>(raw), 4)) return false;
    value = static_cast<std::uint32_t>(raw[0])
          | (static_cast<std::uint32_t>(raw[1]) << 8)
          | (static_cast<std::uint32_t>(raw[2]) << 16)
          | (static_cast<std::uint32_t>(raw[3]) << 24);
    return true;
}

inline std::string resolveGalleryFile(const std::string& path) {
    if (path.empty()) return path;
    gallery_fs::path given(path);
    if (gallery_fs::exists(given)) return given.string();
    gallery_fs::path rooted = gallery_fs::path(::dxapp::RuntimeRoot()) / given;
    if (gallery_fs::exists(rooted)) return rooted.string();
    return path;
}

/// Bytes left in `fh` from its read position; 0 when it cannot tell.
inline std::uint64_t bytesLeft(std::ifstream& fh) {
    const std::streampos here = fh.tellg();
    if (here < 0) return 0;
    fh.seekg(0, std::ios::end);
    const std::streampos end = fh.tellg();
    fh.seekg(here);
    return end > here ? static_cast<std::uint64_t>(end - here) : 0;
}

inline bool readBlob(std::ifstream& fh, std::string& out) {
    std::uint32_t length = 0;
    if (!readU32(fh, length)) return false;
    if (length > bytesLeft(fh)) return false;   // a crafted length must not allocate past the file
    out.assign(length, '\0');
    return length == 0 || static_cast<bool>(fh.read(&out[0], length));
}

}  // namespace gallery_detail

/// Load a gallery .bin. Never throws: a missing file is a missing comparison, not a
/// wrong one, and the visualizer says so in the picture.
inline RetrievalGallery loadRetrievalGallery(const std::string& path) {
    RetrievalGallery g;
    std::size_t slash = path.find_last_of("/\\");
    g.name = (slash == std::string::npos) ? path : path.substr(slash + 1);

    const std::string resolved = gallery_detail::resolveGalleryFile(path);
    std::ifstream fh(resolved, std::ios::binary);
    if (!fh.is_open()) { g.error = "gallery not found: " + path; return g; }

    char magic[8] = {0};
    if (!fh.read(magic, 8) || std::memcmp(magic, "DXGAL1\0", 7) != 0) {
        g.error = "not a DXGAL1 gallery: " + path;
        return g;
    }
    std::uint32_t n = 0, dim = 0;
    if (!gallery_detail::readU32(fh, n) || !gallery_detail::readU32(fh, dim)
        || n == 0 || dim == 0) {
        g.error = "malformed header: " + path;
        return g;
    }

    // The header's n x dim against the file before anything that size is
    // allocated: a truncated or crafted file fails here, not in resize().
    if (static_cast<std::uint64_t>(n) * dim > gallery_detail::bytesLeft(fh) / sizeof(float)) {
        g.error = "truncated embeddings: " + path;
        return g;
    }
    g.vectors.resize(static_cast<std::size_t>(n) * dim);
    if (!fh.read(reinterpret_cast<char*>(g.vectors.data()),
                 static_cast<std::streamsize>(g.vectors.size() * sizeof(float)))) {
        g.error = "truncated embeddings: " + path;
        return g;
    }
    std::string paths_blob, labels_blob;
    if (!gallery_detail::readBlob(fh, g.model)
        || !gallery_detail::readBlob(fh, paths_blob)
        || !gallery_detail::readBlob(fh, labels_blob)) {
        g.error = "truncated metadata: " + path;
        return g;
    }
    g.paths = gallery_detail::splitLines(paths_blob);
    g.labels = gallery_detail::splitLines(labels_blob);
    if (g.paths.size() != n) {
        g.error = "gallery declares " + std::to_string(n) + " entries but carries "
                + std::to_string(g.paths.size()) + " paths: " + path;
        return g;
    }
    g.labels.resize(n);

    // Normalise defensively: the builder writes unit rows, but a hand-edited gallery
    // would otherwise skew every score with no visible symptom.
    for (std::uint32_t i = 0; i < n; ++i) {
        float* row = g.vectors.data() + static_cast<std::size_t>(i) * dim;
        double sum = 0.0;
        for (std::uint32_t d = 0; d < dim; ++d) sum += static_cast<double>(row[d]) * row[d];
        const float norm = static_cast<float>(std::sqrt(sum));
        if (norm > 1e-12f) for (std::uint32_t d = 0; d < dim; ++d) row[d] /= norm;
    }

    g.count = static_cast<int>(n);
    g.dim = static_cast<int>(dim);
    g.loaded = true;
    return g;
}

class GalleryRetrievalPostprocessor : public IPostprocessor<EmbeddingResult> {
public:
    GalleryRetrievalPostprocessor(int input_width, int input_height,
                                  std::string gallery_path = std::string(),
                                  int top_k = 5,
                                  std::string model_name = std::string(),
                                  bool allow_model_mismatch = false)
        : input_width_(input_width), input_height_(input_height),
          gallery_path_(std::move(gallery_path)), top_k_(top_k),
          model_name_(std::move(model_name)),
          allow_model_mismatch_(allow_model_mismatch) {}

    std::vector<EmbeddingResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        (void)ctx;
        if (outputs.empty()) return {};
        auto output = outputs[0];
        const float* data = static_cast<const float*>(output->data());
        if (!data) return {};

        auto shape = output->shape();
        int dimension = 1;
        std::size_t offset = 0;
        if (shape.size() == 3 && shape[0] == 1) {
            // [1, seq, D]: pool the last position, matching the Python tree.
            dimension = static_cast<int>(shape[2]);
            offset = static_cast<std::size_t>(shape[1] - 1) * dimension;
        } else {
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i == 0 && shape.size() > 1 && shape[0] == 1) continue;
                dimension *= static_cast<int>(shape[i]);
            }
        }

        EmbeddingResult result;
        result.embedding.assign(data + offset, data + offset + dimension);
        result.dimension = dimension;

        double sum = 0.0;
        for (float v : result.embedding) sum += static_cast<double>(v) * v;
        const float norm = static_cast<float>(std::sqrt(sum));
        if (norm > 1e-8f) for (float& v : result.embedding) v /= norm;

        const RetrievalGallery& g = gallery();
        if (g.empty()) return {result};
        if (!checkSpace(g, dimension)) return {result};

        std::vector<float> scores(static_cast<std::size_t>(g.count), 0.0f);
        for (int i = 0; i < g.count; ++i) {
            const float* row = g.vectors.data() + static_cast<std::size_t>(i) * g.dim;
            float dot = 0.0f;
            for (int d = 0; d < g.dim; ++d) dot += row[d] * result.embedding[d];
            scores[static_cast<std::size_t>(i)] = dot;
        }

        std::vector<int> order(static_cast<std::size_t>(g.count));
        std::iota(order.begin(), order.end(), 0);
        const int k = std::min(top_k_, g.count);
        // stable_sort, and index as the tie-break, so the ranking is reproducible and
        // matches numpy's stable argsort in the Python tree.
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (scores[static_cast<std::size_t>(a)] != scores[static_cast<std::size_t>(b)])
                return scores[static_cast<std::size_t>(a)] > scores[static_cast<std::size_t>(b)];
            return a < b;
        });

        result.matches.reserve(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) {
            const int j = order[static_cast<std::size_t>(i)];
            GalleryMatch m;
            m.rank = i + 1;
            m.score = scores[static_cast<std::size_t>(j)];
            m.path = g.paths[static_cast<std::size_t>(j)];
            m.label = g.labels[static_cast<std::size_t>(j)];
            result.matches.push_back(std::move(m));
        }
        result.gallery_size = g.count;
        result.gallery_name = g.name;
        return {result};
    }

    std::string getModelName() const override {
        return model_name_.empty() ? std::string("gallery_retrieval") : model_name_;
    }

    /// Why no gallery is loaded, for the visualizer to display. Empty when fine.
    const std::string& galleryError() const { return gallery().error; }

private:
    const RetrievalGallery& gallery() const {
        if (!tried_) {
            tried_ = true;
            if (gallery_path_.empty()) gallery_.error = "no gallery configured";
            else gallery_ = loadRetrievalGallery(gallery_path_);
        }
        return gallery_;
    }

    bool checkSpace(const RetrievalGallery& g, int dimension) const {
        if (g.dim != dimension) {
            std::cerr << "[DXAPP] [ERROR] GalleryRetrievalPostprocessor - gallery '"
                      << g.name << "' is " << g.dim << "-d and this model emits "
                      << dimension << "-d. A gallery belongs to the model that built "
                      << "it; rebuild it with scripts/build_gallery_database.py --variant "
                      << getModelName() << std::endl;
            return false;
        }
        if (!g.model.empty() && !model_name_.empty() && g.model != model_name_
            && !allow_model_mismatch_) {
            std::cerr << "[DXAPP] [ERROR] GalleryRetrievalPostprocessor - gallery '"
                      << g.name << "' was built by '" << g.model << "' but this is '"
                      << model_name_ << "'. Two encoders of the same width still "
                      << "occupy different embedding spaces, so the ranking would be "
                      << "meaningless while still looking plausible." << std::endl;
            return false;
        }
        return true;
    }

    int input_width_;
    int input_height_;
    std::string gallery_path_;
    int top_k_;
    std::string model_name_;
    bool allow_model_mismatch_;
    mutable RetrievalGallery gallery_;
    mutable bool tried_{false};
};

}  // namespace dxapp

#endif  // GALLERY_RETRIEVAL_POSTPROCESSOR_HPP
