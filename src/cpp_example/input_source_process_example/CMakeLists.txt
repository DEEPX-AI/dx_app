cmake_minimum_required(VERSION 3.16)
project(cpp_example DESCRIPTION "DEEPX C++ Example For Input Source Processing")

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_subdirectory(camera)
add_subdirectory(image)
add_subdirectory(rtsp)
add_subdirectory(video)
