//
// Created by Jesse Qu on 2019-08-02.
//

#include "py_enums.h"

void py_bind_enums(py::module &m) {
    py::enum_<PaddleDType>(m, "PaddleDType", py::arithmetic())
            .value("FLOAT32", PaddleDType::FLOAT32)
            .value("FLOAT16", PaddleDType::FLOAT16)
            .value("INT64",   PaddleDType::INT64  )
            .value("INT8",    PaddleDType::INT8   );

    py::enum_<LayoutType>(m, "LayoutType", py::arithmetic())
            .value("LAYOUT_HWC", LayoutType::LAYOUT_HWC)
            .value("LAYOUT_CHW", LayoutType::LAYOUT_CHW);

    py::enum_<PaddleEngineKind>(m, "PaddleEngineKind", py::arithmetic())
            .value("kPaddleMobile", PaddleEngineKind::kPaddleMobile);
}

