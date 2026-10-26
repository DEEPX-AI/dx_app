#include "ppmatting_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

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

}  // namespace

PPMattingPostProcess::PPMattingPostProcess(int input_w, int input_h)
    : input_width_(input_w), input_height_(input_h) {}

PPMattingResult PPMattingPostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    // Whatever single-channel layout the map arrives in, the H x W plane is
    // contiguous: (1, 1, H, W) and (1, H, W, 1) hold the same bytes.
    dxrt::TensorPtr alpha;
    int height = 0, width = 0;
    for (const auto& o : outputs) {
        const auto& s = o->shape();
        if (s.size() == 4 && s[0] == 1) {
            if (s[1] == 1) { alpha = o; height = static_cast<int>(s[2]); width = static_cast<int>(s[3]); }
            else if (s[3] == 1) { alpha = o; height = static_cast<int>(s[1]); width = static_cast<int>(s[2]); }
        } else if (s.size() == 3 && s[0] == 1) {
            alpha = o; height = static_cast<int>(s[1]); width = static_cast<int>(s[2]);
        } else if (s.size() == 2) {
            alpha = o; height = static_cast<int>(s[0]); width = static_cast<int>(s[1]);
        }
        if (alpha) break;
    }
    if (!alpha) {
        std::ostringstream msg;
        msg << "[DXAPP] [ERROR] PPMattingPostProcess - no single-channel alpha map found in "
               "the model outputs.\n  Got: " << format_shapes(outputs) << "\n"
            << "  A matting model emits one channel: (1, 1, H, W), (1, H, W, 1) or (1, H, W). "
               "A multi-class segmentation head is refused here on purpose -- silently taking "
               "its channel 0 would yield a believable but wrong matte. Use SemanticSegPostProcess "
               "for a class map.";
        throw std::runtime_error(msg.str());
    }
    if (height <= 0 || width <= 0) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] PPMattingPostProcess - the alpha map is empty: " +
            format_shapes(outputs));
    }
    if (alpha->type() != dxrt::DataType::FLOAT) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] PPMattingPostProcess - expected a float32 alpha map, got " +
            format_shapes(outputs));
    }

    const std::size_t count = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    const float* data = static_cast<const float*>(alpha->data());
    PPMattingResult result;
    result.alpha.assign(data, data + count);
    result.width = width;
    result.height = height;

    // A model compiled without the final sigmoid emits logits; squash those and
    // leave a map that already is a probability alone.
    bool logits = false;
    for (float v : result.alpha) {
        if (v < 0.0f || v > 1.0f) { logits = true; break; }
    }
    if (logits) {
        for (float& v : result.alpha) v = 1.0f / (1.0f + std::exp(-v));
    }
    for (float& v : result.alpha) v = std::min(1.0f, std::max(0.0f, v));
    return result;
}
