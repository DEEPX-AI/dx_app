#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <dxrt/dxrt_api.h>
#include "yolov5_postprocess.h"
#include "yolov5_ppu_postprocess.h"
#include "yolov7_postprocess.h"
#include "yolov7_ppu_postprocess.h"
#include "yolov8_postprocess.h"
#include "yolov8seg_postprocess.h"
#include "yolov9_postprocess.h"
#include "yolov11_postprocess.h"
#include "yolov12_postprocess.h"
#include "yolox_postprocess.h"
#include "yolov5face_postprocess.h"
#include "yolov5pose_postprocess.h"
#include "yolov5pose_ppu_postprocess.h"
#include "scrfd_postprocess.h"
#include "scrfd_ppu_postprocess.h"
#include "deeplabv3_postprocess.h"
#include "yolov26_postprocess.h"
#include "yolov10_postprocess.h"
#include "semantic_seg_postprocess.h"
#include "classification_postprocess.h"
#include "depth_postprocess.h"
#include "yolov8pose_postprocess.h"
#include "ssd_postprocess.h"
#include "nanodet_postprocess.h"
#include "damoyolo_postprocess.h"
#include "obb_postprocess.h"
#include "dncnn_postprocess.h"
#include "yolov5seg_postprocess.h"
#include "embedding_postprocess.h"
#include "espcn_postprocess.h"
#include "zero_dce_postprocess.h"
#include "face3d_postprocess.h"
#include "retinaface_postprocess.h"
#include "ulfgfd_postprocess.h"
#include "centerpose_postprocess.h"

namespace py = pybind11;

dxrt::TensorPtrs numpy_to_dxrt_tensors(py::list ie_output, const std::vector<dxrt::DataType>& force_dtypes = {}) {
    const size_t num_tensors = ie_output.size();
    dxrt::TensorPtrs tensors;
    tensors.reserve(num_tensors);
    
    for (size_t i = 0; i < num_tensors; ++i) {
        py::array output_arr = py::cast<py::array>(ie_output[i]);
        py::buffer_info info = output_arr.request();
        
        dxrt::DataType dtype;
        if (i < force_dtypes.size()) {
            dtype = force_dtypes[i];
        } else if (info.format == py::format_descriptor<float>::format()) {
            dtype = dxrt::DataType::FLOAT;
        } else if (info.format == py::format_descriptor<int32_t>::format()) {
            dtype = dxrt::DataType::INT32;
        } else if (info.format == py::format_descriptor<int64_t>::format()) {
            dtype = dxrt::DataType::INT64;
        } else if (info.format == py::format_descriptor<uint8_t>::format()) {
            dtype = dxrt::DataType::UINT8;
        } else {
            throw std::runtime_error("Unsupported data type in numpy array: " + info.format);
        }
        
        const size_t num_dims = info.shape.size();
        std::vector<int64_t> shape;
        shape.reserve(num_dims);
        for (size_t j = 0; j < num_dims; ++j) {
            shape.emplace_back(static_cast<int64_t>(info.shape[j]));
        }
        
        auto tensor = std::make_shared<dxrt::Tensor>(
            "output_" + std::to_string(i),
            std::move(shape),
            dtype,
            info.ptr
        );
        
        tensors.emplace_back(std::move(tensor));
    }
    
    return tensors;
}

py::array_t<float> yolov5_results_to_numpy(const std::vector<YOLOv5Result>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolov7_results_to_numpy(const std::vector<YOLOv7Result>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolov8_results_to_numpy(const std::vector<YOLOv8Result>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::tuple yolov8seg_results_to_numpy(const std::vector<YOLOv8SegResult>& results) {
    const size_t num_results = results.size();
    py::array_t<float> detections(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto det_buf = detections.mutable_unchecked<2>();

    if (num_results == 0) {
        py::array_t<uint8_t> empty_masks(std::vector<py::ssize_t>{0, 0, 0});
        return py::make_tuple(detections, empty_masks);
    }

    int mask_h = results[0].mask_height;
    int mask_w = results[0].mask_width;
    py::array_t<uint8_t> masks(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), mask_h, mask_w});
    auto mask_buf = masks.mutable_unchecked<3>();

    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        if (result.box.size() >= 4) {
            det_buf(i, 0) = result.box[0];
            det_buf(i, 1) = result.box[1];
            det_buf(i, 2) = result.box[2];
            det_buf(i, 3) = result.box[3];
        } else {
            det_buf(i, 0) = det_buf(i, 1) = det_buf(i, 2) = det_buf(i, 3) = 0.0f;
        }
        det_buf(i, 4) = result.confidence;
        det_buf(i, 5) = static_cast<float>(result.class_id);

        if (!result.mask.empty() && result.mask_height > 0 && result.mask_width > 0 &&
            static_cast<int>(result.mask.size()) == result.mask_height * result.mask_width) {
            for (int h = 0; h < result.mask_height; ++h) {
                for (int w = 0; w < result.mask_width; ++w) {
                    float v = result.mask[h * result.mask_width + w];
                    if (v < 0.0f) v = 0.0f;
                    else if (v > 1.0f) v = 1.0f;
                    uint8_t mv = static_cast<uint8_t>(v * 255.0f);
                    mask_buf(i, h, w) = mv;
                }
            }
        } else {
            for (int h = 0; h < mask_h; ++h) {
                for (int w = 0; w < mask_w; ++w) {
                    mask_buf(i, h, w) = 0;
                }
            }
        }
    }

    return py::make_tuple(detections, masks);
}

py::array_t<float> yolov9_results_to_numpy(const std::vector<YOLOv9Result>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolov11_results_to_numpy(const std::vector<YOLOv11Result>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolov12_results_to_numpy(const std::vector<YOLOv12Result>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolox_results_to_numpy(const std::vector<YOLOXResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolov5face_results_to_numpy(const std::vector<YOLOv5FaceResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 21}); // 6 (box+conf+class) + 15 (5 landmarks * 3)
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 21});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = 0.0f; // face class_id is always 0

        // Add landmarks (5 points * 2 coordinates = 10 values)
        for (size_t j = 0; j < 10 && j < result.landmarks.size(); ++j) {
            buf(i, 6 + j) = result.landmarks[j];
        }
    }
    
    return detections;
}

py::array_t<float> yolov5pose_results_to_numpy(const std::vector<YOLOv5PoseResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 57}); // 6 (box+conf+class) + 51 (17 landmarks * 3)
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 57});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = 0.0f; // person class_id is always 0
        
        // Add landmarks (17 points * 3 coordinates = 51 values)
        for (size_t j = 0; j < 51 && j < result.landmarks.size(); ++j) {
            buf(i, 6 + j) = result.landmarks[j];
        }
    }
    
    return detections;
}

py::array_t<float> scrfd_results_to_numpy(const std::vector<SCRFDResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 21}); // 6 (box+conf+class) + 15 (5 landmarks * 3)
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 21});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = 0.0f; // face class_id is always 0
        
        // Add landmarks (5 points * 2 coordinates = 10 values)
        for (size_t j = 0; j < 10 && j < result.landmarks.size(); ++j) {
            buf(i, 6 + j) = result.landmarks[j];
        }
    }
    
    return detections;
}

py::array_t<int> deeplabv3_result_to_numpy(const DeepLabv3Result& result) {
    py::array_t<int> class_map({result.height, result.width});
    auto buf = class_map.mutable_unchecked<2>();

    for (int h = 0; h < result.height; ++h) {
        for (int w = 0; w < result.width; ++w) {
            buf(h, w) = result.segmentation_mask[h * result.width + w];
        }
    }

    return class_map;
}

py::array_t<float> scrfd_ppu_results_to_numpy(const std::vector<SCRFDPPUResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 21}); // 6 (box+conf+class) + 15 (5 landmarks * 3)
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 21});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = 0.0f; // face class_id is always 0
        
        // Add landmarks (5 points * 2 coordinates = 10 values)
        for (size_t j = 0; j < 10 && j < result.landmarks.size(); ++j) {
            buf(i, 6 + j) = result.landmarks[j];
        }
    }
    
    return detections;
}

py::array_t<float> yolov5_ppu_results_to_numpy(const std::vector<YOLOv5PPUResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolov7_ppu_results_to_numpy(const std::vector<YOLOv7PPUResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolov5pose_ppu_results_to_numpy(const std::vector<YOLOv5PosePPUResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 57}); // 6 (box+conf+class) + 51 (17 landmarks * 3)
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 57});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = 0.0f; // person class_id is always 0
        
        // Add landmarks (17 points * 3 coordinates = 51 values)
        for (size_t j = 0; j < 51 && j < result.landmarks.size(); ++j) {
            buf(i, 6 + j) = result.landmarks[j];
        }
    }
    
    return detections;
}

py::array_t<float> yolov26_results_to_numpy(const std::vector<YOLOv26Result>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> yolov10_results_to_numpy(const std::vector<YOLOv10Result>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<int> semantic_seg_result_to_numpy(const SemanticSegResult& result) {
    py::array_t<int> class_map({result.height, result.width});
    auto buf = class_map.mutable_unchecked<2>();

    for (int h = 0; h < result.height; ++h) {
        for (int w = 0; w < result.width; ++w) {
            buf(h, w) = result.segmentation_mask[h * result.width + w];
        }
    }

    return class_map;
}

py::array_t<float> classification_results_to_numpy(const std::vector<ClassificationResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 2});
    }
    
    py::array_t<float> preds(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 2});
    auto buf = preds.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        buf(i, 0) = static_cast<float>(results[i].class_id);
        buf(i, 1) = results[i].confidence;
    }
    
    return preds;
}

py::array_t<uint8_t> depth_result_to_numpy(const DepthResult& result) {
    py::array_t<uint8_t> depth_map({result.height, result.width});
    auto buf = depth_map.mutable_unchecked<2>();

    for (int h = 0; h < result.height; ++h) {
        for (int w = 0; w < result.width; ++w) {
            buf(h, w) = result.depth_map[h * result.width + w];
        }
    }

    return depth_map;
}

py::array_t<float> yolov8pose_results_to_numpy(const std::vector<YOLOv8PoseResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 57});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 57});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = 0.0f;
        
        for (size_t j = 0; j < 51 && j < result.landmarks.size(); ++j) {
            buf(i, 6 + j) = result.landmarks[j];
        }
    }
    
    return detections;
}

py::array_t<float> ssd_results_to_numpy(const std::vector<SSDResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> nanodet_results_to_numpy(const std::vector<NanoDetResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> damoyolo_results_to_numpy(const std::vector<DamoYOLOResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.box[0];
        buf(i, 1) = result.box[1];
        buf(i, 2) = result.box[2];
        buf(i, 3) = result.box[3];
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
    }
    
    return detections;
}

py::array_t<float> obb_results_to_numpy(const std::vector<OBBResult>& results) {
    const size_t num_results = results.size();
    if (num_results == 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, 7});
    }
    
    py::array_t<float> detections(std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 7});
    auto buf = detections.mutable_unchecked<2>();
    
    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        buf(i, 0) = result.cx;
        buf(i, 1) = result.cy;
        buf(i, 2) = result.width;
        buf(i, 3) = result.height;
        buf(i, 4) = result.confidence;
        buf(i, 5) = static_cast<float>(result.class_id);
        buf(i, 6) = result.angle;
    }
    
    return detections;
}

// --- DnCNN result converter ---
py::array_t<float> dncnn_result_to_numpy(const DnCNNResult& result) {
    py::array_t<float> image({result.height, result.width});
    auto buf = image.mutable_unchecked<2>();

    for (int h = 0; h < result.height; ++h) {
        for (int w = 0; w < result.width; ++w) {
            buf(h, w) = result.image[h * result.width + w];
        }
    }

    return image;
}

// --- YOLOv5Seg results converter ---
py::tuple yolov5seg_results_to_numpy(const std::vector<YOLOv5SegResult>& results) {
    const size_t num_results = results.size();
    py::array_t<float> detections(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto det_buf = detections.mutable_unchecked<2>();

    if (num_results == 0) {
        py::array_t<uint8_t> empty_masks(std::vector<py::ssize_t>{0, 0, 0});
        return py::make_tuple(detections, empty_masks);
    }

    int mask_h = results[0].mask_height;
    int mask_w = results[0].mask_width;
    py::array_t<uint8_t> masks(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), mask_h, mask_w});
    auto mask_buf = masks.mutable_unchecked<3>();

    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        if (result.box.size() >= 4) {
            det_buf(i, 0) = result.box[0];
            det_buf(i, 1) = result.box[1];
            det_buf(i, 2) = result.box[2];
            det_buf(i, 3) = result.box[3];
        } else {
            det_buf(i, 0) = det_buf(i, 1) = det_buf(i, 2) = det_buf(i, 3) = 0.0f;
        }
        det_buf(i, 4) = result.confidence;
        det_buf(i, 5) = static_cast<float>(result.class_id);

        if (!result.mask.empty() && result.mask_height > 0 && result.mask_width > 0 &&
            static_cast<int>(result.mask.size()) == result.mask_height * result.mask_width) {
            for (int h = 0; h < result.mask_height; ++h) {
                for (int w = 0; w < result.mask_width; ++w) {
                    float v = result.mask[h * result.mask_width + w];
                    if (v < 0.0f) v = 0.0f;
                    else if (v > 1.0f) v = 1.0f;
                    mask_buf(i, h, w) = static_cast<uint8_t>(v * 255.0f);
                }
            }
        } else {
            for (int h = 0; h < mask_h; ++h) {
                for (int w = 0; w < mask_w; ++w) {
                    mask_buf(i, h, w) = 0;
                }
            }
        }
    }

    return py::make_tuple(detections, masks);
}

// --- Embedding result converter ---
py::array_t<float> embedding_result_to_numpy(const EmbeddingResult& result) {
    py::array_t<float> embedding(result.dimension);
    auto buf = embedding.mutable_unchecked<1>();
    for (int i = 0; i < result.dimension; ++i) {
        buf(i) = result.embedding[i];
    }
    return embedding;
}

// --- ESPCN result converter ---
py::array_t<float> espcn_result_to_numpy(const ESPCNResult& result) {
    if (result.channels == 1) {
        py::array_t<float> image({result.height, result.width});
        auto buf = image.mutable_unchecked<2>();
        for (int h = 0; h < result.height; ++h) {
            for (int w = 0; w < result.width; ++w) {
                buf(h, w) = result.image[h * result.width + w];
            }
        }
        return image;
    } else {
        py::array_t<float> image({result.channels, result.height, result.width});
        auto buf = image.mutable_unchecked<3>();
        for (int c = 0; c < result.channels; ++c) {
            for (int h = 0; h < result.height; ++h) {
                for (int w = 0; w < result.width; ++w) {
                    buf(c, h, w) = result.image[c * result.height * result.width + h * result.width + w];
                }
            }
        }
        return image;
    }
}

// --- Zero-DCE result converter ---
py::array_t<float> zero_dce_result_to_numpy(const ZeroDCEResult& result) {
    py::array_t<float> image({result.channels, result.height, result.width});
    auto buf = image.mutable_unchecked<3>();
    for (int c = 0; c < result.channels; ++c) {
        for (int h = 0; h < result.height; ++h) {
            for (int w = 0; w < result.width; ++w) {
                buf(c, h, w) = result.image[c * result.height * result.width + h * result.width + w];
            }
        }
    }
    return image;
}

// --- Face3D result converter ---
py::array_t<float> face3d_result_to_numpy(const Face3DResult& result) {
    py::array_t<float> params(result.num_params);
    auto buf = params.mutable_unchecked<1>();
    for (int i = 0; i < result.num_params; ++i) {
        buf(i) = result.params[i];
    }
    return params;
}

// ─────────────────────────────────────────────────────────────────────────────
// RetinaFace:  [N, 16] = box(4) + conf(1) + cls(1) + 5 landmarks × 2
// ─────────────────────────────────────────────────────────────────────────────
py::array_t<float> retinaface_results_to_numpy(const std::vector<RetinaFaceResult>& results) {
    const size_t N = results.size();
    if (N == 0) return py::array_t<float>(std::vector<py::ssize_t>{0, 16});
    py::array_t<float> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(N), 16});
    auto buf = out.mutable_unchecked<2>();
    for (size_t i = 0; i < N; ++i) {
        const auto& r = results[i];
        buf(i, 0) = r.box[0];
        buf(i, 1) = r.box[1];
        buf(i, 2) = r.box[2];
        buf(i, 3) = r.box[3];
        buf(i, 4) = r.confidence;
        buf(i, 5) = 0.0f;  // face class_id = 0
        for (size_t j = 0; j < 10 && j < r.landmarks.size(); ++j) {
            buf(i, 6 + static_cast<py::ssize_t>(j)) = r.landmarks[j];
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// ULFGFD:  [N, 6] = box(4) + conf(1) + cls(1)   (no landmarks)
// ─────────────────────────────────────────────────────────────────────────────
py::array_t<float> ulfgfd_results_to_numpy(const std::vector<ULFGFDResult>& results) {
    const size_t N = results.size();
    if (N == 0) return py::array_t<float>(std::vector<py::ssize_t>{0, 6});
    py::array_t<float> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(N), 6});
    auto buf = out.mutable_unchecked<2>();
    for (size_t i = 0; i < N; ++i) {
        const auto& r = results[i];
        buf(i, 0) = r.box[0];
        buf(i, 1) = r.box[1];
        buf(i, 2) = r.box[2];
        buf(i, 3) = r.box[3];
        buf(i, 4) = r.confidence;
        buf(i, 5) = 0.0f;  // face class_id = 0
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// CenterPose:  [N, 30] = box(4) + conf(1) + cls(1) + 8 keypoints × 3 (x,y,conf)
// ─────────────────────────────────────────────────────────────────────────────
py::array_t<float> centerpose_results_to_numpy(const std::vector<CenterPoseResult>& results) {
    const size_t N = results.size();
    if (N == 0) return py::array_t<float>(std::vector<py::ssize_t>{0, 30});
    py::array_t<float> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(N), 30});
    auto buf = out.mutable_unchecked<2>();
    for (size_t i = 0; i < N; ++i) {
        const auto& r = results[i];
        buf(i, 0) = r.box[0];
        buf(i, 1) = r.box[1];
        buf(i, 2) = r.box[2];
        buf(i, 3) = r.box[3];
        buf(i, 4) = r.confidence;
        buf(i, 5) = static_cast<float>(r.class_id);
        for (size_t j = 0; j < 24 && j < r.landmarks.size(); ++j) {
            buf(i, 6 + static_cast<py::ssize_t>(j)) = r.landmarks[j];
        }
    }
    return out;
}

PYBIND11_MODULE(dx_postprocess, m)
{
    py::class_<YOLOv5PostProcess>(m, "YOLOv5PostProcess")
        .def(py::init<int, int, float, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("obj_threshold"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv5PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv5Result> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov5_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv5PostProcess::get_input_width)
        .def("get_input_height", &YOLOv5PostProcess::get_input_height);

    py::class_<YOLOv7PostProcess>(m, "YOLOv7PostProcess")
        .def(py::init<int, int, float, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("obj_threshold"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv7PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv7Result> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov7_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv7PostProcess::get_input_width)
        .def("get_input_height", &YOLOv7PostProcess::get_input_height);

    py::class_<YOLOv8PostProcess>(m, "YOLOv8PostProcess")
        .def(py::init<int, int, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv8PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv8Result> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov8_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv8PostProcess::get_input_width)
        .def("get_input_height", &YOLOv8PostProcess::get_input_height);
    
    py::class_<YOLOv8SegPostProcess>(m, "YOLOv8SegPostProcess")
        .def(py::init<int, int, float, float, bool>(),
             py::arg("input_w"),
             py::arg("input_h"),
             py::arg("score_threshold"),
             py::arg("nms_threshold"),
             py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv8SegPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv8SegResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov8seg_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv8SegPostProcess::get_input_width)
        .def("get_input_height", &YOLOv8SegPostProcess::get_input_height);

    py::class_<YOLOv9PostProcess>(m, "YOLOv9PostProcess")
        .def(py::init<int, int, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv9PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv9Result> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov9_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv9PostProcess::get_input_width)
        .def("get_input_height", &YOLOv9PostProcess::get_input_height);
    
    py::class_<YOLOv11PostProcess>(m, "YOLOv11PostProcess")
        .def(py::init<int, int, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv11PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv11Result> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov11_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv11PostProcess::get_input_width)
        .def("get_input_height", &YOLOv11PostProcess::get_input_height);
    
    py::class_<YOLOv12PostProcess>(m, "YOLOv12PostProcess")
        .def(py::init<int, int, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv12PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv12Result> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov12_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv12PostProcess::get_input_width)
        .def("get_input_height", &YOLOv12PostProcess::get_input_height);

    py::class_<YOLOXPostProcess>(m, "YOLOXPostProcess")
        .def(py::init<int, int, float, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("obj_threshold"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOXPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOXResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolox_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOXPostProcess::get_input_width)
        .def("get_input_height", &YOLOXPostProcess::get_input_height);

    py::class_<YOLOv5FacePostProcess>(m, "YOLOv5FacePostProcess")
        .def(py::init<int, int, float, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("obj_threshold"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv5FacePostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv5FaceResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov5face_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv5FacePostProcess::get_input_width)
        .def("get_input_height", &YOLOv5FacePostProcess::get_input_height);

    py::class_<YOLOv5PosePostProcess>(m, "YOLOv5PosePostProcess")
        .def(py::init<int, int, float, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("obj_threshold"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv5PosePostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv5PoseResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov5pose_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv5PosePostProcess::get_input_width)
        .def("get_input_height", &YOLOv5PosePostProcess::get_input_height);

    py::class_<SCRFDPostProcess>(m, "SCRFDPostProcess")
        .def(py::init<int, int, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](SCRFDPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<SCRFDResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return scrfd_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &SCRFDPostProcess::get_input_width)
        .def("get_input_height", &SCRFDPostProcess::get_input_height);

    py::class_<DeepLabv3PostProcess>(m, "DeepLabv3PostProcess")
        .def(py::init<int, int>(),
            py::arg("input_w"),
            py::arg("input_h"))
        .def("postprocess", [](DeepLabv3PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            DeepLabv3Result result;
            {
                py::gil_scoped_release release;
                result = self.postprocess(tensors);
            }
            return deeplabv3_result_to_numpy(result);
        }, py::arg("ie_output"))
        .def("get_input_width", &DeepLabv3PostProcess::get_input_width)
        .def("get_input_height", &DeepLabv3PostProcess::get_input_height);

    py::class_<YOLOv5PPUPostProcess>(m, "YOLOv5PPUPostProcess")
        .def(py::init<int, int, float, float, float>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("obj_threshold"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"))
        .def("postprocess", [](YOLOv5PPUPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output, {dxrt::DataType::BBOX});
            std::vector<YOLOv5PPUResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov5_ppu_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv5PPUPostProcess::get_input_width)
        .def("get_input_height", &YOLOv5PPUPostProcess::get_input_height);

    py::class_<YOLOv7PPUPostProcess>(m, "YOLOv7PPUPostProcess")
        .def(py::init<int, int, float, float, float>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("obj_threshold"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"))
        .def("postprocess", [](YOLOv7PPUPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output, {dxrt::DataType::BBOX});
            std::vector<YOLOv7PPUResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov7_ppu_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv7PPUPostProcess::get_input_width)
        .def("get_input_height", &YOLOv7PPUPostProcess::get_input_height);

    py::class_<YOLOv5PosePPUPostProcess>(m, "YOLOv5PosePPUPostProcess")
        .def(py::init<int, int, float, float>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"))
        .def("postprocess", [](YOLOv5PosePPUPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output, {dxrt::DataType::POSE});
            std::vector<YOLOv5PosePPUResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov5pose_ppu_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv5PosePPUPostProcess::get_input_width)
        .def("get_input_height", &YOLOv5PosePPUPostProcess::get_input_height);

    py::class_<SCRFDPPUPostProcess>(m, "SCRFDPPUPostProcess")
        .def(py::init<int, int, float, float>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"))
        .def("postprocess", [](SCRFDPPUPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output, {dxrt::DataType::FACE});
            std::vector<SCRFDPPUResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return scrfd_ppu_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &SCRFDPPUPostProcess::get_input_width)
        .def("get_input_height", &SCRFDPPUPostProcess::get_input_height);

    // --- YOLOv26 ---
    py::class_<YOLOv26PostProcess>(m, "YOLOv26PostProcess")
        .def(py::init<int, int, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv26PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv26Result> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov26_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv26PostProcess::get_input_width)
        .def("get_input_height", &YOLOv26PostProcess::get_input_height);

    // --- YOLOv10 ---
    py::class_<YOLOv10PostProcess>(m, "YOLOv10PostProcess")
        .def(py::init<int, int, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv10PostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv10Result> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov10_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv10PostProcess::get_input_width)
        .def("get_input_height", &YOLOv10PostProcess::get_input_height);

    // --- SemanticSeg (generic) ---
    py::class_<SemanticSegPostProcess>(m, "SemanticSegPostProcess")
        .def(py::init<int, int, int>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("num_classes") = 0)
        .def("postprocess", [](SemanticSegPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            SemanticSegResult result;
            {
                py::gil_scoped_release release;
                result = self.postprocess(tensors);
            }
            return semantic_seg_result_to_numpy(result);
        }, py::arg("ie_output"))
        .def("get_input_width", &SemanticSegPostProcess::get_input_width)
        .def("get_input_height", &SemanticSegPostProcess::get_input_height);

    // --- Classification ---
    py::class_<ClassificationPostProcess>(m, "ClassificationPostProcess")
        .def(py::init<int>(),
            py::arg("top_k") = 5)
        .def("postprocess", [](ClassificationPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<ClassificationResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return classification_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_top_k", &ClassificationPostProcess::get_top_k);

    // --- Depth ---
    py::class_<DepthPostProcess>(m, "DepthPostProcess")
        .def(py::init<int, int>(),
            py::arg("input_w"),
            py::arg("input_h"))
        .def("postprocess", [](DepthPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            DepthResult result;
            {
                py::gil_scoped_release release;
                result = self.postprocess(tensors);
            }
            return depth_result_to_numpy(result);
        }, py::arg("ie_output"))
        .def("get_input_width", &DepthPostProcess::get_input_width)
        .def("get_input_height", &DepthPostProcess::get_input_height);

    // --- YOLOv8Pose ---
    py::class_<YOLOv8PosePostProcess>(m, "YOLOv8PosePostProcess")
        .def(py::init<int, int, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured") = false)
        .def("postprocess", [](YOLOv8PosePostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv8PoseResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov8pose_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv8PosePostProcess::get_input_width)
        .def("get_input_height", &YOLOv8PosePostProcess::get_input_height);

    // --- SSD ---
    py::class_<SSDPostProcess>(m, "SSDPostProcess")
        .def(py::init<int, int, float, float, int, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("num_classes") = 20,
            py::arg("has_background") = true)
        .def("postprocess", [](SSDPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<SSDResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return ssd_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &SSDPostProcess::get_input_width)
        .def("get_input_height", &SSDPostProcess::get_input_height);

    // --- NanoDet ---
    py::class_<NanoDetPostProcess>(m, "NanoDetPostProcess")
        .def(py::init<int, int, float, float, int, int>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("num_classes") = 80,
            py::arg("reg_max") = 10)
        .def("postprocess", [](NanoDetPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<NanoDetResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return nanodet_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &NanoDetPostProcess::get_input_width)
        .def("get_input_height", &NanoDetPostProcess::get_input_height);

    // --- DamoYOLO ---
    py::class_<DamoYOLOPostProcess>(m, "DamoYOLOPostProcess")
        .def(py::init<int, int, float, float, int>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("num_classes") = 80)
        .def("postprocess", [](DamoYOLOPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<DamoYOLOResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return damoyolo_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &DamoYOLOPostProcess::get_input_width)
        .def("get_input_height", &DamoYOLOPostProcess::get_input_height);

    // --- OBB ---
    py::class_<OBBPostProcess>(m, "OBBPostProcess")
        .def(py::init<int, int, float>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold") = 0.3f)
        .def("postprocess", [](OBBPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<OBBResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return obb_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &OBBPostProcess::get_input_width)
        .def("get_input_height", &OBBPostProcess::get_input_height);

    // --- DnCNN ---
    py::class_<DnCNNPostProcess>(m, "DnCNNPostProcess")
        .def(py::init<int, int>(),
            py::arg("input_w"),
            py::arg("input_h"))
        .def("postprocess", [](DnCNNPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            DnCNNResult result;
            {
                py::gil_scoped_release release;
                result = self.postprocess(tensors);
            }
            return dncnn_result_to_numpy(result);
        }, py::arg("ie_output"))
        .def("get_input_width", &DnCNNPostProcess::get_input_width)
        .def("get_input_height", &DnCNNPostProcess::get_input_height);

    // --- YOLOv5Seg ---
    py::class_<YOLOv5SegPostProcess>(m, "YOLOv5SegPostProcess")
        .def(py::init<int, int, float, float, float, bool>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("obj_threshold"),
            py::arg("score_threshold"),
            py::arg("nms_threshold"),
            py::arg("is_ort_configured") = true)
        .def("postprocess", [](YOLOv5SegPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<YOLOv5SegResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return yolov5seg_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width", &YOLOv5SegPostProcess::get_input_width)
        .def("get_input_height", &YOLOv5SegPostProcess::get_input_height)
        .def("get_obj_threshold", &YOLOv5SegPostProcess::get_obj_threshold)
        .def("get_score_threshold", &YOLOv5SegPostProcess::get_score_threshold)
        .def("get_nms_threshold", &YOLOv5SegPostProcess::get_nms_threshold);

    // --- Embedding (CLIP, ArcFace) ---
    py::class_<EmbeddingPostProcess>(m, "EmbeddingPostProcess")
        .def(py::init<bool>(),
            py::arg("l2_normalize") = true)
        .def("postprocess", [](EmbeddingPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            EmbeddingResult result;
            {
                py::gil_scoped_release release;
                result = self.postprocess(tensors);
            }
            return embedding_result_to_numpy(result);
        }, py::arg("ie_output"))
        .def("get_l2_normalize", &EmbeddingPostProcess::get_l2_normalize);

    // --- ESPCN (Super Resolution) ---
    py::class_<ESPCNPostProcess>(m, "ESPCNPostProcess")
        .def(py::init<int, int, int>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("scale_factor") = 2)
        .def("postprocess", [](ESPCNPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            ESPCNResult result;
            {
                py::gil_scoped_release release;
                result = self.postprocess(tensors);
            }
            return espcn_result_to_numpy(result);
        }, py::arg("ie_output"))
        .def("get_input_width", &ESPCNPostProcess::get_input_width)
        .def("get_input_height", &ESPCNPostProcess::get_input_height)
        .def("get_scale_factor", &ESPCNPostProcess::get_scale_factor);

    // --- Zero-DCE (Image Enhancement) ---
    py::class_<ZeroDCEPostProcess>(m, "ZeroDCEPostProcess")
        .def(py::init<int, int>(),
            py::arg("input_w"),
            py::arg("input_h"))
        .def("postprocess", [](ZeroDCEPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            ZeroDCEResult result;
            {
                py::gil_scoped_release release;
                result = self.postprocess(tensors);
            }
            return zero_dce_result_to_numpy(result);
        }, py::arg("ie_output"))
        .def("get_input_width", &ZeroDCEPostProcess::get_input_width)
        .def("get_input_height", &ZeroDCEPostProcess::get_input_height);

    // --- Face3D (3DDFA v2 Face Alignment) ---
    py::class_<Face3DPostProcess>(m, "Face3DPostProcess")
        .def(py::init<int, int>(),
            py::arg("input_w"),
            py::arg("input_h"))
        .def("postprocess", [](Face3DPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            Face3DResult result;
            {
                py::gil_scoped_release release;
                result = self.postprocess(tensors);
            }
            return face3d_result_to_numpy(result);
        }, py::arg("ie_output"))
        .def("get_input_width", &Face3DPostProcess::get_input_width)
        .def("get_input_height", &Face3DPostProcess::get_input_height);

    // --- RetinaFace ---
    py::class_<RetinaFacePostProcess>(m, "RetinaFacePostProcess")
        .def(py::init<int, int, float, float>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold") = 0.5f,
            py::arg("nms_threshold")   = 0.4f)
        .def("postprocess", [](RetinaFacePostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<RetinaFaceResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return retinaface_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width",  &RetinaFacePostProcess::get_input_width)
        .def("get_input_height", &RetinaFacePostProcess::get_input_height);

    // --- ULFGFD ---
    py::class_<ULFGFDPostProcess>(m, "ULFGFDPostProcess")
        .def(py::init<int, int, float, float>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold") = 0.7f,
            py::arg("nms_threshold")   = 0.3f)
        .def("postprocess", [](ULFGFDPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<ULFGFDResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return ulfgfd_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width",  &ULFGFDPostProcess::get_input_width)
        .def("get_input_height", &ULFGFDPostProcess::get_input_height);

    // --- CenterPose ---
    py::class_<CenterPosePostProcess>(m, "CenterPosePostProcess")
        .def(py::init<int, int, float, float, int>(),
            py::arg("input_w"),
            py::arg("input_h"),
            py::arg("score_threshold") = 0.3f,
            py::arg("nms_threshold")   = 0.5f,
            py::arg("num_keypoints")   = 8)
        .def("postprocess", [](CenterPosePostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);
            std::vector<CenterPoseResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            return centerpose_results_to_numpy(results);
        }, py::arg("ie_output"))
        .def("get_input_width",  &CenterPosePostProcess::get_input_width)
        .def("get_input_height", &CenterPosePostProcess::get_input_height);
}