//
// Created by Jesse Qu on 2019-08-02.
//

#include "py_paddlebuf.h"

void py_bind_paddlebuf(py::module &m) {
    py::class_<PaddleBuf>(m, "PaddleBuf")
        .def(py::init<>())
        .def(py::init<const PaddleBuf&>())
        .def(py::init([](py::ssize_t& size) {
            auto ptr = new char[size];
            memset(ptr, 0x00, size * sizeof(char));
            return new PaddleBuf(ptr, size * sizeof(char));
        }))
        .def(py::init([](py::array_t<float>& input) {
            auto buf = input.request();
            return new PaddleBuf(buf.ptr, buf.size * sizeof(float));
        }))
        .def("__repr__", [](const PaddleBuf& pb) {
            auto *data = (float *)pb.data();
            std::ostringstream out;
            out << "<paddlemobile.PaddleBuf object at " << &pb << " (with size ";
            out << std::dec << pb.length() << ")>" << std::endl;
            out << "buf: [ ";
            for (size_t i = 0; i < pb.length()/sizeof(float); i++) {
                out << std::dec << data[i] << " ";
            }
            out << "]" << std::endl;
            out << "data count: " << std::dec << pb.length()/sizeof(float);
            return out.str();
        })
        .def("empty", &PaddleBuf::empty)
        .def("length", &PaddleBuf::length);
}
