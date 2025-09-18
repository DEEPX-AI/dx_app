cmake_minimum_required(VERSION 3.14)

if(MSVC)
    if(NOT DEFINED DXRT_DIR OR NOT EXISTS ${DXRT_DIR})
        message(FATAL_ERROR "The variable DXRT_DIR is not set. The build cannot proceed.")
    endif()
    string(REPLACE "\\" "/" DXRT_DIR ${DXRT_DIR})
    message(STATUS "The variable DXRT_DIR is set: ${DXRT_DIR}")
    set(DXRT_INSTALLED_DIR ${DXRT_DIR})
    if(NOT DEFINED OpenCV_DIR OR NOT EXISTS ${OpenCV_DIR})
        message(OpenCV_DIR "The variable OpenCV_DIR is not set. The build cannot proceed.")
    endif()
    string(REPLACE "\\" "/" OpenCV_DIR ${OpenCV_DIR})
    message(STATUS "The variable OpenCV_DIR is set: ${OpenCV_DIR}")
endif()

project(dxapp DESCRIPTION "DEEPX Application Templates")

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR} ABSOLUTE)
add_compile_definitions(PROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")

if (MSVC)
    if (DEFINED ENV{WindowsSdkDir})
        set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{WindowsSdkDir}")
    else()
        set(CMAKE_SYSTEM_PREFIX_PATH "C:/Program Files (x86)/Windows Kits/10")
    endif()
endif()

# check vaapi
execute_process(
    COMMAND gst-inspect-1.0 vaapidecodebin
    RESULT_VARIABLE GST_VAAPI_FOUND
    OUTPUT_QUIET ERROR_QUIET
)

if(GST_VAAPI_FOUND EQUAL 0)
    message(STATUS "Found vaapi: Defining USE_VAAPI")
    add_definitions(-DUSE_VAAPI)
else()
    message(STATUS "vaapi not found")
endif()

set(CMAKE_CXX_STANDARD 14)
if(MSVC)
    set(CMAKE_CXX_STANDARD 17)
 endif()
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/release" CACHE PATH "Default install path" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) # using function & cfg cmake file

if(NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "${CMAKE_SYSTEM_PROCESSOR}")
    set(CROSS_COMPILE TRUE)
else()
    set(CROSS_COMPILE FALSE)
endif()

set(STD_FS_NO_LIB_NEEDED FALSE)
if(CMAKE_CXX_STANDARD GREATER 17 OR CMAKE_CXX_STANDARD EQUAL 17)
    set(STD_FS_NO_LIB_NEEDED TRUE)
elseif(MSVC)
    set(STD_FS_NO_LIB_NEEDED TRUE)
endif()

include(dxapp.cfg)
include(dxapp.function)

if(CROSS_COMPILE)
    set(CMAKE_SKIP_RPATH TRUE)
    file(READ "${DXRT_INSTALLED_DIR}/include/dxrt/gen.h" header_contents)
    if(header_contents MATCHES "#define USE_ORT")
        message(STATUS "THIS IS ONNX_RUNTIME")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${onnxruntime_LIB_DIRS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link,${onnxruntime_LIB_DIRS}")
    endif()
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MP /wd4127")
    add_compile_options(-D_AMD64_ -D_WIN64 -DWIN64)
    add_compile_options(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
else()
    add_compile_options(-W -Wall -pthread -fstrict-volatile-bitfields -fPIC)
endif()

if(ENABLE_DEBUG_INFO)
    if (NOT MSVC) 
        add_compile_options(-g -rdynamic)
    endif()
endif()

if(MSVC)
    execute_process(
        COMMAND where python
        OUTPUT_VARIABLE PYTHON_PATHS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REPLACE "\n" ";" PYTHON_PATH_LIST "${PYTHON_PATHS}")
    list(GET PYTHON_PATH_LIST 0 Python_EXECUTABLE)
    message(STATUS "Selected Python Executable : ${Python_EXECUTABLE}")
else()
    execute_process(
        COMMAND which python
        OUTPUT_VARIABLE Python_EXECUTABLE 
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

add_subdirectory(templates/classification)
add_subdirectory(templates/object_detection)

add_subdirectory(demos/object_detection)
add_subdirectory(demos/classification)
add_subdirectory(demos/pose_estimation)
add_subdirectory(demos/object_det_and_seg)
add_subdirectory(demos/segmentation)

if(EXISTS "${CMAKE_SOURCE_DIR}/test" AND USE_DXAPP_TEST)
    add_subdirectory(test)
endif()

message(STATUS "Compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")
message(STATUS "Using c++ standard required: ${CMAKE_CXX_STANDARD_REQUIRED}")
message(STATUS "Using c++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}") 
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "DXRT_INSTALLED_DIR: ${DXRT_INSTALLED_DIR}")