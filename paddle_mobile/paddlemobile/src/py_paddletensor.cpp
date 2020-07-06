//
// Created by Jesse Qu on 2019-08-02.
//

#include "py_paddletensor.h"

void py_bind_paddletensor(py::module &m) {
    py::class_<PaddleTensor>(m, "PaddleTensor", py::buffer_protocol())
        .def(py::init<>())
        .def_readwrite("name"  , &PaddleTensor::name  )
        .def_readwrite("shape" , &PaddleTensor::shape )
        .def_readwrite("dtype" , &PaddleTensor::dtype )
        .def_readwrite("layout", &PaddleTensor::layout)
        .def("get_shape", [](const PaddleTensor &pt) -> py::tuple {
            py::tuple t(pt.shape.size());
            for (size_t i = 0; i < pt.shape.size(); i++) {
                t[i] = pt.shape[i];
            }
            return t;
        })
        .def("set_shape", [](PaddleTensor &pt, const py::tuple& shape) -> void {
            pt.shape.clear();
            for (const auto & i : shape) {
                pt.shape.push_back(i.cast<int>());
            }
        })
        .def_readwrite("data", &PaddleTensor::data)
        .def_buffer([](PaddleTensor &pt) -> py::buffer_info {
            std::vector<int> strides;
            strides.resize(pt.shape.size());
            auto rit_shape = pt.shape.rbegin();
            auto rit_strides = strides.rbegin();
            *rit_strides = sizeof(float);
            rit_strides++;
            for (; rit_strides != strides.rend(); ++rit_strides, ++rit_shape) {
                *rit_strides = (*(rit_strides - 1)) * (*rit_shape);
            }
            return py::buffer_info(
                (float *)pt.data.data(),
                pt.shape,
                strides
            );
        });
}
