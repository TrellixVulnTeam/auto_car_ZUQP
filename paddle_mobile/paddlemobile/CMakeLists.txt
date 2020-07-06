cmake_minimum_required(VERSION 2.8.12)
project(paddlemobile)

set(PADDLE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set(PADDLE_LIB_DIR "${PROJECT_SOURCE_DIR}/lib" )

include_directories(${PADDLE_INCLUDE_DIR})
link_directories(${PADDLE_LIB_DIR})

set(PROJECT_SOURCES src/pybind.h src/pybind.cpp
        src/py_enums.h src/py_enums.cpp
        src/py_paddlebuf.h src/py_paddlebuf.cpp
        src/py_paddletensor.h src/py_paddletensor.cpp
        src/py_paddlemobileconfig.h src/py_paddlemobileconfig.cpp
        src/py_paddlepredicator.h src/py_paddlepredicator.cpp)

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} ${PROJECT_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE paddle-mobile)
