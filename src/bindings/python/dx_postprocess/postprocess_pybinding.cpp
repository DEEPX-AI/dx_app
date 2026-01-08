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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));
    
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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));
    
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
        }, py::arg("ie_output"));
    
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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));

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
        }, py::arg("ie_output"));
}