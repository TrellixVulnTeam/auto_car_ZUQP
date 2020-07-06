//
// Created by Jesse Qu on 2019-08-02.
//

#ifndef PADDLEMOBILE_PYBIND_H
#define PADDLEMOBILE_PYBIND_H

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "io/paddle_inference_api.h"

namespace py = pybind11;
using namespace paddle_mobile;

#endif //PADDLEMOBILE_PYBIND_H
