#include "anomaly_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

std::string format_shapes(const dxrt::TensorPtrs& outputs) {
    std::ostringstream s;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        s << (i ? ", " : "") << "(";
        const auto& shape = outputs[i]->shape();
        for (std::size_t d = 0; d < shape.size(); ++d) {
            s << (d ? ", " : "") << shape[d];
        }
        s << ")";
    }
    return s.str();
}

// One float feature map, either layout. The spatial grid of these networks is
// square (64x64 at 256x256, 28x28 at 224x224), which is what tells the layouts
// apart: (1, H, W, C) has the equal pair FIRST, (1, C, H, W) has it LAST.
// Channels-first wins when all three are equal, as in the Python postprocessors.
struct FeatureMap {
    const float* data{nullptr};
    int channels{0};
    int height{0};
    int width{0};
    bool channels_last{false};

    float at(int c, int y, int x) const {
        const std::ptrdiff_t idx =
            channels_last
                ? (static_cast<std::ptrdiff_t>(y) * width + x) * channels + c
                : (static_cast<std::ptrdiff_t>(c) * height + y) * width + x;
        return data[idx];
    }
};

bool to_feature_map(const dxrt::TensorPtr& tensor, FeatureMap* out) {
    const auto& s = tensor->shape();
    if (s.size() != 4 || s[0] != 1 || tensor->type() != dxrt::DataType::FLOAT) {
        return false;
    }
    const int a = static_cast<int>(s[1]);
    const int b = static_cast<int>(s[2]);
    const int c = static_cast<int>(s[3]);
    out->data = static_cast<const float*>(tensor->data());
    out->channels_last = (a == b && b != c);
    out->channels = out->channels_last ? c : a;
    out->height = out->channels_last ? a : b;
    out->width = out->channels_last ? b : c;
    return true;
}

// Frame-relative 0..1. A flat response stays at 0 rather than becoming all-hot.
std::vector<float> normalise(const std::vector<float>& response) {
    const auto mm = std::minmax_element(response.begin(), response.end());
    const float low = *mm.first;
    const float span = *mm.second - low;
    std::vector<float> out(response.size(), 0.0f);
    if (span > 1e-12f) {
        for (std::size_t i = 0; i < response.size(); ++i) {
            out[i] = (response[i] - low) / span;
        }
    }
    return out;
}

// numpy.percentile rejects q outside [0, 100] with ValueError; a q past 100 would
// index past the end below. std::invalid_argument becomes ValueError in Python.
float checked_percentile(const char* owner, float q) {
    if (!(q >= 0.0f && q <= 100.0f)) {  // also rejects NaN
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] " << owner << " - score_percentile must be in [0, 100], got "
            << q;
        throw std::invalid_argument(msg.str());
    }
    return q;
}

// numpy.percentile(values, q) with its default linear interpolation; q in [0, 100].
float linear_percentile(std::vector<float> values, double q) {
    if (values.empty()) return 0.0f;
    const double pos = (q / 100.0) * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    std::nth_element(values.begin(), values.begin() + lo, values.end());
    const double low = values[lo];
    if (hi == lo) return static_cast<float>(low);
    const double high = *std::min_element(values.begin() + lo + 1, values.end());
    return static_cast<float>(low + (high - low) * (pos - static_cast<double>(lo)));
}

}  // namespace

EfficientADPostProcess::EfficientADPostProcess(std::vector<std::string> roles,
                                               float st_weight, float ae_weight,
                                               float score_percentile)
    : roles_(std::move(roles)),
      st_weight_(st_weight),
      ae_weight_(ae_weight),
      score_percentile_(checked_percentile("EfficientADPostProcess", score_percentile)) {}

AnomalyMapResult EfficientADPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    if (outputs.size() != 3) {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] EfficientADPostProcess - needs all three EfficientAD "
               "networks, got "
            << outputs.size() << " output(s): " << format_shapes(outputs) << "\n"
            << "  Expected a 768-channel student plus a 384-channel teacher and a "
               "384-channel autoencoder.";
        throw std::runtime_error(msg.str());
    }

    std::vector<FeatureMap> maps(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (!to_feature_map(outputs[i], &maps[i])) {
            throw std::runtime_error(
                "[DXAPP] [ERROR] EfficientADPostProcess - expected float (1, C, H, W) "
                "feature maps, got " + format_shapes(outputs));
        }
    }

    const FeatureMap* student = nullptr;
    const FeatureMap* teacher = nullptr;
    const FeatureMap* autoencoder = nullptr;
    if (roles_.size() == maps.size()) {
        for (std::size_t i = 0; i < maps.size(); ++i) {
            if (roles_[i] == "student") student = &maps[i];
            else if (roles_[i] == "teacher") teacher = &maps[i];
            else if (roles_[i] == "autoencoder") autoencoder = &maps[i];
        }
        if (student == nullptr || teacher == nullptr || autoencoder == nullptr) {
            std::ostringstream msg;
            msg << "[DXAPP] [ERROR] EfficientADPostProcess - roles do not name every "
                   "network:";
            for (const auto& r : roles_) msg << " " << r;
            throw std::runtime_error(msg.str());
        }
    } else {
        std::size_t widest = 0;
        for (std::size_t i = 1; i < maps.size(); ++i) {
            if (maps[i].channels > maps[widest].channels) widest = i;
        }
        student = &maps[widest];
        for (std::size_t i = 0; i < maps.size(); ++i) {
            if (i == widest) continue;
            if (teacher == nullptr) teacher = &maps[i];
            else autoencoder = &maps[i];
        }
    }

    if (student->channels != teacher->channels + autoencoder->channels ||
        teacher->channels != autoencoder->channels ||
        teacher->height != student->height || teacher->width != student->width ||
        autoencoder->height != student->height || autoencoder->width != student->width) {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] EfficientADPostProcess - the student does not split into "
               "its two targets: student " << student->channels << "x" << student->height
            << "x" << student->width << ", teacher " << teacher->channels << "x"
            << teacher->height << "x" << teacher->width << ", autoencoder "
            << autoencoder->channels << "x" << autoencoder->height << "x"
            << autoencoder->width;
        throw std::runtime_error(msg.str());
    }

    const int half = teacher->channels;
    const int h = teacher->height;
    const int w = teacher->width;
    std::vector<float> st(static_cast<std::size_t>(h) * w);
    std::vector<float> ae(st.size());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum_st = 0.0;
            double sum_ae = 0.0;
            for (int c = 0; c < half; ++c) {
                const double d_st = teacher->at(c, y, x) - student->at(c, y, x);
                const double d_ae = autoencoder->at(c, y, x) - student->at(c + half, y, x);
                sum_st += d_st * d_st;
                sum_ae += d_ae * d_ae;
            }
            const std::size_t i = static_cast<std::size_t>(y) * w + x;
            st[i] = static_cast<float>(sum_st / half);
            ae[i] = static_cast<float>(sum_ae / half);
        }
    }

    // The map is normalised per frame so a hot region shows whatever the
    // magnitudes; the score reads the raw disagreement so frames can be ranked.
    const std::vector<float> st_norm = normalise(st);
    const std::vector<float> ae_norm = normalise(ae);
    std::vector<float> raw(st.size());
    AnomalyMapResult result;
    result.anomaly_map.resize(st.size());
    for (std::size_t i = 0; i < st.size(); ++i) {
        result.anomaly_map[i] = st_weight_ * st_norm[i] + ae_weight_ * ae_norm[i];
        raw[i] = st_weight_ * st[i] + ae_weight_ * ae[i];
    }
    result.width = w;
    result.height = h;
    result.score = linear_percentile(std::move(raw), score_percentile_);
    result.channels = student->channels;
    return result;
}

AnomalyFeaturePostProcess::AnomalyFeaturePostProcess(float score_percentile)
    : score_percentile_(checked_percentile("AnomalyFeaturePostProcess", score_percentile)) {}

AnomalyMapResult AnomalyFeaturePostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    FeatureMap features;
    bool found = false;
    for (const auto& tensor : outputs) {
        if (to_feature_map(tensor, &features)) {
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] AnomalyFeaturePostProcess - no float (1, C, H, W) feature "
            "map in the model outputs: " + format_shapes(outputs));
    }

    // Per-pixel L2 norm across channels: the magnitude of the feature response,
    // which is what every distance-based anomaly method reduces to per location.
    const int h = features.height;
    const int w = features.width;
    std::vector<float> response(static_cast<std::size_t>(h) * w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;
            for (int c = 0; c < features.channels; ++c) {
                const double v = features.at(c, y, x);
                sum += v * v;
            }
            response[static_cast<std::size_t>(y) * w + x] = static_cast<float>(std::sqrt(sum));
        }
    }

    AnomalyMapResult result;
    result.anomaly_map = normalise(response);
    result.width = w;
    result.height = h;
    result.score = linear_percentile(result.anomaly_map, score_percentile_);
    result.channels = features.channels;
    return result;
}
