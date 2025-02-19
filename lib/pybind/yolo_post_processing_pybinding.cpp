#include <pybind11/pybind11.h>
#include "yolo_post_processing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dx_postprocess, m)
{
    py::class_<YoloPostProcess>(m, "YoloPostProcess")
        .def(py::init<py::dict>())
        .def("Run", &YoloPostProcess::Run);
}