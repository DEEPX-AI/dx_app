cmake_minimum_required(VERSION 3.16)
project(dx_postprocess_module)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
if(MSVC)
    # Resolve DXRT install location from environment (set by build_env.bat)
    set(DXRT_INSTALLED_DIR "$ENV{DXRT_INSTALLED_DIR}")
    if(NOT DXRT_INSTALLED_DIR)
        message(FATAL_ERROR "DXRT_INSTALLED_DIR is not set. Run build_env.bat before building dx_postprocess.")
    endif()
    message(STATUS "DXRT_INSTALLED_DIR: ${DXRT_INSTALLED_DIR}")

    # Locate DXRT headers and library
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
else()
    find_library(DXRT_LIB dxrt REQUIRED)
endif()

# Set PROJECT_ROOT
if(NOT DEFINED ENV{PROJECT_ROOT})
    get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
    message(STATUS "Auto-detected PROJECT_ROOT: ${PROJECT_ROOT}")
else()
    set(PROJECT_ROOT $ENV{PROJECT_ROOT})
    message(STATUS "Using PROJECT_ROOT from environment: ${PROJECT_ROOT}")
endif()
if(MSVC)
    string(REPLACE "\\" "/" PROJECT_ROOT ${PROJECT_ROOT})
    string(REPLACE "\\" "/" DXRT_INSTALLED_DIR ${DXRT_INSTALLED_DIR})
    set(INCLUDE_DIRS ${DXRT_INSTALLED_DIR}/include)
endif()

# Find pybind11 (pip-installed)
find_package(pybind11 CONFIG REQUIRED)

set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
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
    "${PROJECT_ROOT}/src/postprocess/yolov11"
    "${PROJECT_ROOT}/src/postprocess/yolov12"
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

# Use the correct extension per platform so Python can import the module
if(MSVC)
    # Windows Python extension modules use .pyd
    set_target_properties(dx_postprocess PROPERTIES PREFIX "" SUFFIX ".pyd")
else()
    set_target_properties(dx_postprocess PROPERTIES PREFIX "" SUFFIX ".so")
endif()

install(TARGETS dx_postprocess DESTINATION .)

if(DEFINED ENV{DEBUG})
    if(MSVC)
        target_compile_options(dx_postprocess PRIVATE /Od /Zi /W3)
    else()
        target_compile_options(dx_postprocess PRIVATE -g -O0 -Wall)
    endif()
    target_compile_definitions(dx_postprocess PRIVATE DX_DEBUG)
    message(STATUS "Building dx_postprocess in DEBUG mode")
else()
    if(MSVC)
        target_compile_options(dx_postprocess PRIVATE /O2 /W3)
    else()
        target_compile_options(dx_postprocess PRIVATE -O3 -march=native -Wall)
    endif()
    target_compile_definitions(dx_postprocess PRIVATE NDEBUG)
endif()

if(MSVC)
    # Silence <experimental/filesystem> deprecation warning until code is migrated
    target_compile_definitions(dx_postprocess PRIVATE _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
endif()