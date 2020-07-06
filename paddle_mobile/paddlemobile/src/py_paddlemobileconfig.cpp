//
// Created by Jesse Qu on 2019-08-08.
//

#include "py_paddlemobileconfig.h"

void py_bind_paddlemobileconfig(py::module &m) {
//    py::enum_<PaddleMobileConfig::Precision>(m, "Precision", py::arithmetic())
//        .value("FP32", PaddleMobileConfig::Precision::FP32);

//    py::enum_<PaddleMobileConfig::Device>(m, "Device", py::arithmetic())
//        .value("kCPU",      PaddleMobileConfig::Device::kCPU)
//        .value("kFPGA",     PaddleMobileConfig::Device::kFPGA)
//        .value("kGPU_MALI", PaddleMobileConfig::Device::kGPU_MALI)
//        .value("kGPU_CL",   PaddleMobileConfig::Device::kGPU_CL);

    py::class_<PaddleMobileConfig> config(m, "PaddleMobileConfig");
        py::enum_<PaddleMobileConfig::Precision>(config, "Precision", py::arithmetic())
            .value("FP32", PaddleMobileConfig::Precision::FP32);

        py::enum_<PaddleMobileConfig::Device>(config, "Device", py::arithmetic())
            .value("kCPU",      PaddleMobileConfig::Device::kCPU)
            .value("kFPGA",     PaddleMobileConfig::Device::kFPGA)
            .value("kGPU_MALI", PaddleMobileConfig::Device::kGPU_MALI)
            .value("kGPU_CL",   PaddleMobileConfig::Device::kGPU_CL);

        config
            .def(py::init<>())
            .def(py::init<const PaddleMobileConfig&>())
            .def_readwrite("precision",      &PaddleMobileConfig::precision)
            .def_readwrite("device",         &PaddleMobileConfig::device)
            .def_readwrite("model_dir",      &PaddleMobileConfig::model_dir)
            .def_readwrite("prog_file",      &PaddleMobileConfig::prog_file)
            .def_readwrite("param_file",     &PaddleMobileConfig::param_file)
            .def_readwrite("thread_num",     &PaddleMobileConfig::thread_num)
            .def_readwrite("batch_size",     &PaddleMobileConfig::batch_size)
            .def_readwrite("lod_mode",       &PaddleMobileConfig::lod_mode)
            .def_readwrite("optimize",       &PaddleMobileConfig::optimize)
            .def_readwrite("quantification", &PaddleMobileConfig::quantification);
}
