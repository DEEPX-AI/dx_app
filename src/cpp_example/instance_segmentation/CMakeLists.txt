cmake_minimum_required(VERSION 3.14)
project(cpp_example DESCRIPTION "DEEPX C++ Example For Object Detection Programs")

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_subdirectory(yolov8seg)