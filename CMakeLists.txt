cmake_minimum_required(VERSION 3.14)
get_filename_component(PROJECT_ROOT ${CMAKE_SOURCE_DIR} ABSOLUTE)
project(dxapp DESCRIPTION "DEEPX Application Templates")

if(MSVC)
    if(NOT DEFINED DXRT_DIR OR NOT EXISTS ${DXRT_DIR})
        message(FATAL_ERROR "The variable DXRT_DIR is not set. The build cannot proceed.")
    endif()
    if(NOT DEFINED DXRT_INSTALLED_DIR)
    set(DXRT_INSTALLED_DIR "${DXRT_DIR}")
    message(STATUS "DXRT_INSTALLED_DIR is set to ${DXRT_INSTALLED_DIR}")
    endif()
    string(REPLACE "\\" "/" DXRT_DIR ${DXRT_DIR})
    message(STATUS "The variable DXRT_DIR is set: ${DXRT_DIR}")
    if(NOT DEFINED OpenCV_DIR OR NOT EXISTS ${OpenCV_DIR})
        message(OpenCV_DIR "The variable OpenCV_DIR is not set. The build cannot proceed.")
    endif()
    string(REPLACE "\\" "/" OpenCV_DIR ${OpenCV_DIR})
    message(STATUS "The variable OpenCV_DIR is set: ${OpenCV_DIR}")
    set(DXAPP_WITH_SHAREDLIB FALSE)
endif()

add_compile_definitions(PROJECT_ROOT_DIR="${PROJECT_ROOT}")

# Provide portable thread target; on MSVC supply a stub for existing pthread links
find_package(Threads REQUIRED)
if(MSVC AND NOT TARGET pthread)
    add_library(pthread INTERFACE)
    target_link_libraries(pthread INTERFACE Threads::Threads)
endif()

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

# Code coverage option
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)

if(ENABLE_COVERAGE AND NOT MSVC)
    message(STATUS "Code coverage enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
elseif(ENABLE_COVERAGE AND MSVC)
    message(WARNING "Code coverage is not supported on MSVC builds")
endif()

# Determine if filesystem library linking is needed
set(STD_FS_NO_LIB_NEEDED FALSE)
if(CMAKE_CXX_STANDARD GREATER 17 OR CMAKE_CXX_STANDARD EQUAL 17)
    if(MSVC)
        # MSVC with C++17 has filesystem in standard library
        set(STD_FS_NO_LIB_NEEDED TRUE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0")
        # GCC 9.0+ with C++17 has filesystem in standard library
        set(STD_FS_NO_LIB_NEEDED TRUE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0")
        # Clang 9.0+ with C++17 has filesystem in standard library
        set(STD_FS_NO_LIB_NEEDED TRUE)
    endif()
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/release" CACHE PATH "Default install path" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) # using function & cfg cmake file

if(NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "${CMAKE_SYSTEM_PROCESSOR}")
    set(CROSS_COMPILE TRUE)
else()
    set(CROSS_COMPILE FALSE)
endif()

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
    add_compile_options(/utf-8)  # Avoid CP949 mis-decoding of UTF-8 sources
    add_definitions(-DNOMINMAX)
else()
    add_compile_options(-W -Wall -Wextra -pthread -O3 -fstrict-volatile-bitfields -fPIC)
endif()

if(ENABLE_DEBUG_INFO)
    message(STATUS "Debug info enabled")
    if (NOT MSVC) 
        add_compile_options(-g -rdynamic)
        message(STATUS "Debug info enabled for g++/clang++")
    
        set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
        message(STATUS "Debug info enabled for sonarlint")
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
endif()

if(DXAPP_WITH_SHAREDLIB)
add_subdirectory(src/postprocess)
endif()
add_subdirectory(src/cpp_example)

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