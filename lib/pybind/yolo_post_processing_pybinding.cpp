#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "yolo_post_processing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dx_postprocess, m)
{
    py::class_<YoloPostProcess>(m, "YoloPostProcess")
        .def(py::init<py::dict>())
        .def("SetConfig", &YoloPostProcess::SetConfig,
            py::arg("config"))
        .def("Run", &YoloPostProcess::Run,
            py::arg("ie_output"),
            py::arg("ratio") = std::make_pair(1.0f, 1.0f),
            py::arg("pad") = std::make_pair(0.0f, 0.0f));
}