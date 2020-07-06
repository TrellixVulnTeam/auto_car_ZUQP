//
// Created by Jesse Qu on 2019-08-04.
//

#include "py_paddlepredicator.h"

void py_bind_paddlepredicator(py::module &m) {
    py::class_<PaddlePredictor>(m, "PaddlePredictor")
        .def("Run", [](PaddlePredictor &predictor,
                       const std::vector<PaddleTensor>& inputs) {
            PaddleTensor tensor_out;
            tensor_out.shape = std::vector<int>({});
            tensor_out.data = PaddleBuf();

            tensor_out.dtype = PaddleDType::FLOAT32;
            auto *outputs = new std::vector<PaddleTensor>(1, tensor_out);

            predictor.Run(inputs, outputs, -1);

            return outputs;
        }, py::return_value_policy::take_ownership);

    m.def("CreatePaddlePredictor", &CreatePaddlePredictor<PaddleMobileConfig, PaddleEngineKind::kPaddleMobile>);
}


