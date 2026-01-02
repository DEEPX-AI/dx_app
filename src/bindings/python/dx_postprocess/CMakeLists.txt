cmake_minimum_required(VERSION 3.16)
project(dx_postprocess_module)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_library(DXRT_LIB dxrt REQUIRED)

# Set PROJECT_ROOT
if(NOT DEFINED ENV{PROJECT_ROOT})
    get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
    message(STATUS "Auto-detected PROJECT_ROOT: ${PROJECT_ROOT}")
else()
    set(PROJECT_ROOT $ENV{PROJECT_ROOT})
    message(STATUS "Using PROJECT_ROOT from environment: ${PROJECT_ROOT}")
endif()

# Find pybind11 (pip-installed)
find_package(pybind11 CONFIG REQUIRED)

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${PROJECT_ROOT}/src/postprocess/deeplabv3"
    "${PROJECT_ROOT}/src/postprocess/scrfd"
    "${PROJECT_ROOT}/src/postprocess/scrfd_ppu"
    "${PROJECT_ROOT}/src/postprocess/yolov5"
    "${PROJECT_ROOT}/src/postprocess/yolov5_ppu"
    "${PROJECT_ROOT}/src/postprocess/yolov5face"
    "${PROJECT_ROOT}/src/postprocess/yolov5pose"
    "${PROJECT_ROOT}/src/postprocess/yolov5pose_ppu"
    "${PROJECT_ROOT}/src/postprocess/yolov7"
    "${PROJECT_ROOT}/src/postprocess/yolov7_ppu"
    "${PROJECT_ROOT}/src/postprocess/yolov8"
    "${PROJECT_ROOT}/src/postprocess/yolov8seg"
    "${PROJECT_ROOT}/src/postprocess/yolov9"
    "${PROJECT_ROOT}/src/postprocess/yolox"
    "${PROJECT_ROOT}/src/utility"
    "${PROJECT_ROOT}/extern"
)

file(GLOB POSTPROCESS_SOURCES "${PROJECT_ROOT}/src/postprocess/*/*.cpp")
set(SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/postprocess_pybinding.cpp"
    ${POSTPROCESS_SOURCES}
    "${PROJECT_ROOT}/src/utility/common_util.cpp"
)

pybind11_add_module(dx_postprocess ${SOURCES})
target_include_directories(dx_postprocess PRIVATE ${INCLUDE_DIRS})
target_link_libraries(dx_postprocess PRIVATE ${DXRT_LIB})

set_target_properties(dx_postprocess PROPERTIES
    PREFIX ""
    SUFFIX ".so")

install(TARGETS dx_postprocess DESTINATION .)

if(DEFINED ENV{DEBUG})
    target_compile_options(dx_postprocess PRIVATE -g -O0 -Wall)
    target_compile_definitions(dx_postprocess PRIVATE DX_DEBUG)
    message(STATUS "Building dx_postprocess in DEBUG mode")
else()
    target_compile_options(dx_postprocess PRIVATE -O3 -march=native -Wall)
    target_compile_definitions(dx_postprocess PRIVATE NDEBUG)
endif()