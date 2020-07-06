cmake_minimum_required(VERSION 3.5.1)
project(edgeboard_samples)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
set(PADDLE_LIB_DIR "${PROJECT_SOURCE_DIR}/lib" )
set(EASYDL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )

set(COMMON_SOURCES
        src/blocking_queue.h
        src/camera.hpp
        src/camera.cpp
        src/json.hpp
        )

find_package(OpenCV QUIET COMPONENTS core videoio highgui imgproc imgcodecs ml video)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
#list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})，${OpenCV_LIBS}")
#add_definitions(-DUSE_OPENCV)

include_directories(${EASYDL_INCLUDE_DIR})
LINK_DIRECTORIES(${PADDLE_LIB_DIR})

set(IMAGE_DETECTION "image_detection" )
add_executable(${IMAGE_DETECTION} ${COMMON_SOURCES} src/image_detection.cpp)
target_link_libraries(${IMAGE_DETECTION} paddle-mobile)
target_link_libraries(${IMAGE_DETECTION} ${OpenCV_LIBS} ) 

set(VIDEO_DETECTION "video_detection" )
add_executable(${VIDEO_DETECTION} ${COMMON_SOURCES} src/video_detection.cpp)
target_link_libraries(${VIDEO_DETECTION} paddle-mobile)
target_link_libraries(${VIDEO_DETECTION} ${OpenCV_LIBS} ) 

