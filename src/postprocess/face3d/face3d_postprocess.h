#ifndef FACE3D_POSTPROCESS_H
#define FACE3D_POSTPROCESS_H

#include <dxrt/dxrt_api.h>
#include <string>
#include <vector>

struct Face3DResult {
    std::vector<float> params{};
    int num_params{0};

    Face3DResult() = default;
    Face3DResult(std::vector<float> p, int n)
        : params(std::move(p)), num_params(n) {}
    ~Face3DResult() = default;
    Face3DResult(const Face3DResult&) = default;
    Face3DResult& operator=(const Face3DResult&) = default;
    Face3DResult(Face3DResult&&) noexcept = default;
    Face3DResult& operator=(Face3DResult&&) noexcept = default;
};

class Face3DPostProcess {
   private:
    int input_width_{120};
    int input_height_{120};

   public:
    Face3DPostProcess(int input_w, int input_h);
    Face3DPostProcess();
    ~Face3DPostProcess() = default;

    Face3DResult postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
};

#endif  // FACE3D_POSTPROCESS_H
