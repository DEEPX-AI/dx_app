cmake_minimum_required(VERSION 3.14)
project(cpp_example DESCRIPTION "DEEPX C++ Example Programs")

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_subdirectory(classification)
add_subdirectory(object_detection)
add_subdirectory(object_detection_x_semantic_segmentation)
add_subdirectory(semantic_segmentation)
add_subdirectory(instance_segmentation)
add_subdirectory(ppu)
