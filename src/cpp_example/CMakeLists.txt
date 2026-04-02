# =============================================================================
# cpp_example - DEEPX C++ Example Programs (Factory Pattern)
# =============================================================================
cmake_minimum_required(VERSION 3.14)
project(cpp_example DESCRIPTION "DEEPX C++ Example Programs")

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# =============================================================================
# Project Root Configuration
# =============================================================================
if(NOT PROJECT_ROOT)
    get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
endif()

set(CPP_EXAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMMON_DIR ${CPP_EXAMPLE_DIR}/common)

# =============================================================================
# Find Required Packages
# =============================================================================
find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR})
find_package(PkgConfig REQUIRED)

# Find dxrt library
if(CROSS_COMPILE OR MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib)
else()
    find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
    set(DXRT_LIB dxrt)
endif()

if(NOT DXRT_LIB)
    message(FATAL_ERROR "dxrt library not found. Please ensure it is installed and accessible.")
endif()

# =============================================================================
# Include Directories
# =============================================================================
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_ROOT}/extern/)
include_directories(${PROJECT_ROOT}/src/utility/)
include_directories(${CPP_EXAMPLE_DIR})  # For common/ includes

# Postprocess headers (header-only .hpp in common/processors/)
include_directories(${CPP_EXAMPLE_DIR}/common/processors)

# =============================================================================
# Common Libraries
# =============================================================================
set(COMMON_LIBS ${OpenCV_LIBS} ${DXRT_LIB})

# Platform-specific filesystem library linking for C++17
if(NOT STD_FS_NO_LIB_NEEDED)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND COMMON_LIBS stdc++fs)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND COMMON_LIBS c++fs)
    endif()
endif()

# =============================================================================
# Common Compiler Flags and Definitions
# =============================================================================
set(COMMON_COMPILE_FLAGS -Wall -Wextra -O3)
set(COMMON_COMPILE_DEFINITIONS PROJECT_ROOT_DIR="${PROJECT_ROOT}")

# =============================================================================
# Common Source Files
# =============================================================================
set(UTILITY_SOURCES
    ${PROJECT_ROOT}/src/utility/common_util.cpp
)

# =============================================================================
# Helper Function to Create Example Executables
# =============================================================================
function(add_example TARGET_NAME SOURCE_FILE POSTPROCESS_DIR)
    add_executable(${TARGET_NAME}
        ${SOURCE_FILE}
        ${UTILITY_SOURCES}
    )

    target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_FLAGS})
    target_compile_definitions(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_DEFINITIONS})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CPP_EXAMPLE_DIR}
    )

    if(DXAPP_WITH_SHAREDLIB)
        target_link_directories(${TARGET_NAME} PRIVATE ${PROJECT_ROOT}/lib/)
        target_link_libraries(${TARGET_NAME} ${COMMON_LIBS})
    else()
        target_link_libraries(${TARGET_NAME} ${COMMON_LIBS})
    endif()
endfunction()

# =============================================================================
# Auto-Discovery: Scan category/model directories for *_sync.cpp & *_async.cpp
# =============================================================================
# Single-model categories (each subdir = one model with {model}_sync.cpp / _async.cpp)
set(CATEGORIES
    object_detection
    classification
    semantic_segmentation
    instance_segmentation
    depth_estimation
    image_denoising
    face_detection
    pose_estimation
    obb_detection
    embedding
    super_resolution
    image_enhancement
    ppu
    face_alignment
    hand_landmark
)

foreach(CATEGORY ${CATEGORIES})
    file(GLOB _MODEL_DIRS RELATIVE ${CPP_EXAMPLE_DIR}/${CATEGORY}
         ${CPP_EXAMPLE_DIR}/${CATEGORY}/*)
    foreach(MODEL ${_MODEL_DIRS})
        set(_MODEL_DIR ${CPP_EXAMPLE_DIR}/${CATEGORY}/${MODEL})
        # Skip non-directories
        if(NOT IS_DIRECTORY ${_MODEL_DIR})
            continue()
        endif()

        if(EXISTS ${_MODEL_DIR}/${MODEL}_sync.cpp)
            add_example(${MODEL}_sync ${_MODEL_DIR}/${MODEL}_sync.cpp "")
        endif()

        if(EXISTS ${_MODEL_DIR}/${MODEL}_async.cpp)
            add_example(${MODEL}_async ${_MODEL_DIR}/${MODEL}_async.cpp "")
            # All async runners use std::thread → need pthread on Linux/GCC
            target_link_libraries(${MODEL}_async pthread)
        endif()
    endforeach()
endforeach()

# =============================================================================
# Multi-Model Examples (category pattern: *_x_*)
# =============================================================================
file(GLOB _MULTI_CATEGORIES RELATIVE ${CPP_EXAMPLE_DIR}
     ${CPP_EXAMPLE_DIR}/*_x_*)
foreach(MULTI_CAT ${_MULTI_CATEGORIES})
    file(GLOB _MULTI_MODELS RELATIVE ${CPP_EXAMPLE_DIR}/${MULTI_CAT}
         ${CPP_EXAMPLE_DIR}/${MULTI_CAT}/*)
    foreach(MODEL ${_MULTI_MODELS})
        set(_MODEL_DIR ${CPP_EXAMPLE_DIR}/${MULTI_CAT}/${MODEL})
        if(NOT IS_DIRECTORY ${_MODEL_DIR})
            continue()
        endif()

        if(EXISTS ${_MODEL_DIR}/${MODEL}_sync.cpp)
            add_example(${MODEL}_sync ${_MODEL_DIR}/${MODEL}_sync.cpp "")
        endif()

        if(EXISTS ${_MODEL_DIR}/${MODEL}_async.cpp)
            add_example(${MODEL}_async ${_MODEL_DIR}/${MODEL}_async.cpp "")
            target_link_libraries(${MODEL}_async pthread)
        endif()
    endforeach()
endforeach()

# =============================================================================
# Installation (Optional)
# =============================================================================
get_property(ALL_TARGETS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
install(TARGETS ${ALL_TARGETS} RUNTIME DESTINATION bin)

