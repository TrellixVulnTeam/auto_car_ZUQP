//
// Created by Jesse Qu on 2019-08-02.
//

#include "pybind.h"
#include "py_enums.h"
#include "py_paddlebuf.h"
#include "py_paddletensor.h"
#include "py_paddlemobileconfig.h"
#include "py_paddlepredicator.h"

PYBIND11_MODULE(paddlemobile, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: paddlemobile

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    py_bind_enums(m);

    py_bind_paddlebuf(m);

    py_bind_paddletensor(m);

    py_bind_paddlemobileconfig(m);

    py_bind_paddlepredicator(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
