#ifndef SFA3D_POSTPROCESS_H
#define SFA3D_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

struct SFA3DResult {
    int class_id{0};
    float confidence{0.0f};
    float x3d{0.0f};
    float y3d{0.0f};
    float z3d{0.0f};
    float dim_h{0.0f};
    float dim_w{0.0f};
    float dim_l{0.0f};
    float yaw{0.0f};
    float bev_x{0.0f};
    float bev_y{0.0f};
    float bev_w{0.0f};
    float bev_h{0.0f};
};

class SFA3DPostProcess {
   public:
    SFA3DPostProcess(int input_w, int input_h,
                     float score_threshold = 0.3f,
                     float nms_threshold = 0.2f);
    SFA3DPostProcess();
    ~SFA3DPostProcess() = default;

    std::vector<SFA3DResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }

   private:
    int input_width_{608};
    int input_height_{608};
    float score_threshold_{0.3f};
    float nms_threshold_{0.2f};
    int max_detections_{100};
};

#endif  // SFA3D_POSTPROCESS_H
