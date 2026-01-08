cmake_minimum_required(VERSION 3.16)
project(object_detection_x_semantic_segmentation DESCRIPTION "DEEPX Multi-Model Examples combining Object Detection and Semantic Segmentation")

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Add multi-model examples
add_subdirectory(yolov7_x_deeplabv3)