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
if(MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib/x64 REQUIRED)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib/x64)
elseif(CROSS_COMPILE)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib)
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
include_directories(${PROJECT_ROOT}/src/postprocess/sfa3d)
include_directories(${PROJECT_ROOT}/src/postprocess/superpoint)
include_directories(${CPP_EXAMPLE_DIR})  # For common/ includes

# Postprocess headers (header-only .hpp in common/processors/)
include_directories(${CPP_EXAMPLE_DIR}/common/processors)

# Postprocess include paths, reused below so a target that does not link
# dxapp_common_obj (its INTERFACE includes) can still compile factories that
# pull in dope/yolopv2/vitpose headers.
set(DXAPP_POSTPROCESS_INCLUDES
    ${PROJECT_ROOT}/src/postprocess/sfa3d
    ${PROJECT_ROOT}/src/postprocess/superpoint
    ${PROJECT_ROOT}/src/postprocess/dope
    ${PROJECT_ROOT}/src/postprocess/yolopv2
    ${PROJECT_ROOT}/src/postprocess/vitpose
)

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
if(MSVC)
    set(COMMON_COMPILE_FLAGS /W3 /MP)
else()
    set(COMMON_COMPILE_FLAGS -Wall -Wextra -O3)
endif()
set(COMMON_COMPILE_DEFINITIONS PROJECT_ROOT_DIR="${PROJECT_ROOT}")
# The graph targets' per-compiler strictness and the codegen's Python check
# (SP6): cmake/dxapp_graph_flags.cmake, cmake/dxapp_codegen_python.cmake.
include(${PROJECT_ROOT}/cmake/dxapp_graph_flags.cmake)
include(${PROJECT_ROOT}/cmake/dxapp_codegen_python.cmake)

# =============================================================================
# Runtime DLL Deployment — Windows/vcpkg only, stage once instead of per target
# =============================================================================
# Windows only. On Linux the loader resolves shared libraries through RPATH and
# ldconfig, there is no vcpkg in the picture (build.sh uses
# cmake/toolchain.<arch>.cmake), and nothing below runs.
if(WIN32)
    # vcpkg keys applocal off the target triplet rather than the compiler, so
    # guard on WIN32 rather than MSVC to also cover clang-cl / MinGW.
    set(VCPKG_APPLOCAL_DEPS OFF)

    if(DEFINED VCPKG_INSTALLED_DIR)
        set(_DXAPP_TRIPLET "${VCPKG_TARGET_TRIPLET}")
        if(NOT _DXAPP_TRIPLET)
            set(_DXAPP_TRIPLET "x64-windows")
        endif()

        # Default RUNTIME_OUTPUT_DIRECTORY layout; nothing here overrides it.
        get_property(_DXAPP_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
        if(_DXAPP_MULTI_CONFIG)
            set(_DXAPP_EXAMPLE_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
        else()
            set(_DXAPP_EXAMPLE_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}")
        endif()

        set(_DXAPP_VCPKG_BIN "$<IF:$<CONFIG:Debug>,\
${VCPKG_INSTALLED_DIR}/${_DXAPP_TRIPLET}/debug/bin,\
${VCPKG_INSTALLED_DIR}/${_DXAPP_TRIPLET}/bin>")

        add_custom_target(dxapp_stage_vcpkg_dlls
            COMMAND ${CMAKE_COMMAND}
                    -DSRC_DIR=${_DXAPP_VCPKG_BIN}
                    -DDST_DIR=${_DXAPP_EXAMPLE_BIN_DIR}
                    -P ${PROJECT_ROOT}/cmake/stage_vcpkg_dlls.cmake
            COMMENT "Staging vcpkg runtime DLLs into the example output directory"
            VERBATIM)
    else()
        message(STATUS "cpp_example: VCPKG_INSTALLED_DIR not set, skipping runtime DLL staging")
    endif()
endif()

# =============================================================================
# Common Source Files — build once as OBJECT library instead of per-target
# =============================================================================
add_library(dxapp_common_obj OBJECT
    ${PROJECT_ROOT}/src/utility/common_util.cpp
    ${PROJECT_ROOT}/src/postprocess/sfa3d/sfa3d_postprocess.cpp
    ${PROJECT_ROOT}/src/postprocess/superpoint/superpoint_postprocess.cpp
    ${PROJECT_ROOT}/src/postprocess/dope/dope_postprocess.cpp
    ${PROJECT_ROOT}/src/postprocess/yolopv2/yolopv2_postprocess.cpp
)
target_compile_options(dxapp_common_obj PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_definitions(dxapp_common_obj PRIVATE ${COMMON_COMPILE_DEFINITIONS})
target_include_directories(dxapp_common_obj
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_ROOT}/extern/
        ${PROJECT_ROOT}/src/utility/
        ${CPP_EXAMPLE_DIR}
        ${CPP_EXAMPLE_DIR}/common/processors
        ${DXAPP_POSTPROCESS_INCLUDES}
    INTERFACE
        ${PROJECT_ROOT}/src/postprocess/superpoint/
        ${PROJECT_ROOT}/src/postprocess/dope/
        ${PROJECT_ROOT}/src/postprocess/yolopv2/
)
# Position-independent so the dx_graph Python module (a shared object) can
# link the same objects the executables do.
set_target_properties(dxapp_common_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Graph engine — the framework's third runner (object library)
# =============================================================================
set(GRAPH_DIR ${COMMON_DIR}/graph)

add_library(dxapp_graph_obj OBJECT
    ${GRAPH_DIR}/shape.cpp
    ${GRAPH_DIR}/graph_config.cpp
    ${GRAPH_DIR}/roi_router.cpp
    ${GRAPH_DIR}/stage_graph.cpp
    ${GRAPH_DIR}/graph_runner_sync.cpp
    ${GRAPH_DIR}/graph_runner_async.cpp
    ${GRAPH_DIR}/graph_visualizer.cpp
    ${GRAPH_DIR}/cpu_reduce.cpp
    ${GRAPH_DIR}/logic_ops.cpp
    ${GRAPH_DIR}/user_logic.cpp
)
target_compile_options(dxapp_graph_obj PRIVATE ${COMMON_COMPILE_FLAGS})
# The engine switches on Shape with no default (SamePayload, ToString):
# a Shape added without its case must fail the build, not
# fall through to a fail-closed return that only a test run would notice.
# -Wreorder is an error here: an initializer list out of declaration order was
# 1044 warning lines per build (SP1 U-13).
# MSVC: /bigobj and its own spellings (cmake/dxapp_graph_flags.cmake).
dxapp_graph_strict_flags(dxapp_graph_obj SWITCH)
target_compile_definitions(dxapp_graph_obj PRIVATE ${COMMON_COMPILE_DEFINITIONS})
target_include_directories(dxapp_graph_obj PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_ROOT}/extern/
    ${PROJECT_ROOT}/src/utility/
    ${CPP_EXAMPLE_DIR}
    ${CPP_EXAMPLE_DIR}/common/processors
    ${DXAPP_POSTPROCESS_INCLUDES}
)
set_target_properties(dxapp_graph_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Generated model registry — the bridge from the 499 per-variant factories to the engine
# =============================================================================
# One translation unit per task, split above 64 models per task. Nothing here
# is hand-maintained: scripts/gen_model_registry.py reads
# config/model_registry.json plus the factory tree and fails the configure
# step on an unparseable header, an ambiguous factory class, a duplicate
# fully qualified factory class (::dxapp::v_<variant>::Cls) or a variant
# header outside its namespace, or a registry entry whose task disagrees with
# its factory's directory.
dxapp_find_codegen_python()

set(GRAPH_GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)

# Re-run the configure step when an input to the codegen changes, so adding
# a model does not need a manual `cmake ..` to make it usable in a graph.
# scripts/add_model.sh always writes config/model_registry.json (and
# scripts/check_model_registry.py enforces the 1:1 with the factory tree), so
# watching that file catches every new model; a factory header edited without
# a registry change is a rename in flight, which `gen_model_registry.py
# --strict` catches in the codegen-strict check of scripts/ci_checks.sh.
# The model-zoo manifest supplies each model's download name
# (ModelInfo::download_name), so a manifest update re-runs the codegen too.
#
# The model table goes into the build tree, never into the source tree: a
# configure must not rewrite a tracked file. The tracked docs/graph_models.md
# is updated with `python3 scripts/gen_model_registry.py --docs-only` and
# guarded by scripts/check_graph_models_doc.py.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${PROJECT_ROOT}/config/model_registry.json
    ${PROJECT_ROOT}/scripts/gen_model_registry.py
    ${PROJECT_ROOT}/scripts/check_factory_uniqueness.py
    ${PROJECT_ROOT}/scripts/modelzoo_manifest.json
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_ROOT}/scripts/gen_model_registry.py
            --out-dir ${GRAPH_GENERATED_DIR}
            --docs ${GRAPH_GENERATED_DIR}/graph_models.md
            --manifest ${PROJECT_ROOT}/scripts/modelzoo_manifest.json
            --print-sources
    OUTPUT_VARIABLE GRAPH_GENERATED_SOURCES
    ERROR_VARIABLE GRAPH_CODEGEN_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GRAPH_CODEGEN_RESULT
)
if(NOT GRAPH_CODEGEN_RESULT EQUAL 0)
    message(FATAL_ERROR
        "gen_model_registry.py failed (${GRAPH_CODEGEN_RESULT}):\n${GRAPH_CODEGEN_STDERR}")
elseif(NOT GRAPH_CODEGEN_STDERR STREQUAL "")
    # A successful run is silent on stderr unless something fell back - an
    # unreadable manifest, an unsafe download name dropped. ERROR_VARIABLE
    # captures that text, so without this line nobody would see it.
    message(WARNING "gen_model_registry.py:\n${GRAPH_CODEGEN_STDERR}")
endif()

# Peak RSS is a few hundred MB per TU and every TU pulls in OpenCV plus its
# task's postprocessors; 25 at once is more memory than a laptop has. Cap the
# concurrency rather than the parallelism of the rest of the build.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS graph_codegen=6)

add_library(dxapp_graph_models_obj OBJECT
    ${CPP_EXAMPLE_DIR}/common/registry/static_model_registry.cpp
    ${GRAPH_GENERATED_SOURCES}
)
set_property(TARGET dxapp_graph_models_obj PROPERTY JOB_POOL_COMPILE graph_codegen)
set_target_properties(dxapp_graph_models_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(dxapp_graph_models_obj PRIVATE ${COMMON_COMPILE_FLAGS})
dxapp_graph_strict_flags(dxapp_graph_models_obj)
target_compile_definitions(dxapp_graph_models_obj PRIVATE ${COMMON_COMPILE_DEFINITIONS})
target_include_directories(dxapp_graph_models_obj PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_ROOT}/extern/
    ${PROJECT_ROOT}/src/utility/
    ${CPP_EXAMPLE_DIR}
    ${CPP_EXAMPLE_DIR}/common/processors
    ${DXAPP_POSTPROCESS_INCLUDES}
)

option(DXAPP_BUILD_GRAPH_TESTS "Build the graph engine test binary" ON)
if(DXAPP_BUILD_GRAPH_TESTS)
    add_executable(graph_engine_test
        ${GRAPH_DIR}/test/graph_engine_test.cpp
    )
    target_compile_options(graph_engine_test PRIVATE ${COMMON_COMPILE_FLAGS})
    # The test fake and the per-shape payload builders switch over enums
    # with no default, so an enumerator added without a matching case is a
    # build error here rather than a warning nobody reads.
    # MSVC: /bigobj and its own spellings (cmake/dxapp_graph_flags.cmake).
    dxapp_graph_strict_flags(graph_engine_test SWITCH)
    target_compile_definitions(graph_engine_test PRIVATE ${COMMON_COMPILE_DEFINITIONS})
    target_include_directories(graph_engine_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_ROOT}/extern/
        ${PROJECT_ROOT}/src/utility/
        ${CPP_EXAMPLE_DIR}
        ${CPP_EXAMPLE_DIR}/common/processors
        ${DXAPP_POSTPROCESS_INCLUDES}
    )
    # dxapp_common_obj carries the out-of-line postprocessors (sfa3d,
    # superpoint, dope, yolopv2) that the generated
    # registry's factories reference. dxapp_graph_consumer_obj carries
    # ReportToJson, which the report tests pin.
    # dxapp_graph_cli_obj: cli::ProjectArgs, an exported project's command line.
    target_link_libraries(graph_engine_test dxapp_graph_cli_obj dxapp_graph_consumer_obj
        dxapp_graph_obj dxapp_graph_models_obj dxapp_common_obj ${COMMON_LIBS})
    if(NOT MSVC)
        target_link_libraries(graph_engine_test pthread)
    endif()
endif()

# Plain-assert tests for common/ code outside the graph engine (processors,
# runner helpers). Same harness as graph_engine_test, driven by
# tests/cpp_example/test_common_unit.py. Header-only code under test, so it
# links OpenCV + dxrt only: no graph objects, no dxapp_common_obj.
option(DXAPP_BUILD_COMMON_TESTS "Build the common/ unit test binary" ON)
if(DXAPP_BUILD_COMMON_TESTS)
    add_executable(common_unit_test
        ${COMMON_DIR}/test/common_unit_test.cpp
        ${PROJECT_ROOT}/src/utility/common_util.cpp
        ${PROJECT_ROOT}/src/postprocess/yolopv2/yolopv2_postprocess.cpp
        ${PROJECT_ROOT}/src/postprocess/superpoint/superpoint_postprocess.cpp
    )
    target_compile_options(common_unit_test PRIVATE ${COMMON_COMPILE_FLAGS})
    target_compile_definitions(common_unit_test PRIVATE ${COMMON_COMPILE_DEFINITIONS})
    target_include_directories(common_unit_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_ROOT}/extern/
        ${PROJECT_ROOT}/src/utility/
        ${CPP_EXAMPLE_DIR}
        ${CPP_EXAMPLE_DIR}/common/processors
        ${DXAPP_POSTPROCESS_INCLUDES}
    )
    target_link_libraries(common_unit_test ${COMMON_LIBS})
    if(NOT MSVC)
        target_link_libraries(common_unit_test pthread)
    endif()
endif()

# =============================================================================
# =============================================================================
# Helper Function to Create Example Executables
# =============================================================================
function(add_example TARGET_NAME SOURCE_FILE POSTPROCESS_DIR)
    add_executable(${TARGET_NAME}
        ${SOURCE_FILE}
    )

    target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_FLAGS})
    target_compile_definitions(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_DEFINITIONS})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CPP_EXAMPLE_DIR}
    )

    if(DXAPP_WITH_SHAREDLIB)
        target_link_directories(${TARGET_NAME} PRIVATE ${PROJECT_ROOT}/lib/)
        target_link_libraries(${TARGET_NAME} dxapp_common_obj ${COMMON_LIBS})
    else()
        target_link_libraries(${TARGET_NAME} dxapp_common_obj ${COMMON_LIBS})
    endif()

    # Windows/vcpkg only: runtime DLLs are staged once up front rather than per
    # target. The target does not exist elsewhere, so this is a no-op on Linux.
    # See the "Runtime DLL Deployment" section above.
    if(TARGET dxapp_stage_vcpkg_dlls)
        add_dependencies(${TARGET_NAME} dxapp_stage_vcpkg_dlls)
    endif()
endfunction()

# =============================================================================
# Multi-model graph example — the CLI over the graph engine
# =============================================================================
# The CLI is the engine's CONSUMER, so it lives OUTSIDE common/graph/: it is
# the one place allowed to name a concrete registry, and
# scripts/check_graph_boundary.py walks every file under common/graph/ to keep
# that name out of the engine. See multi_model_graph/graph_cli.hpp.
set(GRAPH_EXAMPLE_DIR ${CPP_EXAMPLE_DIR}/multi_model_graph)

add_library(dxapp_graph_cli_obj OBJECT
    ${GRAPH_EXAMPLE_DIR}/graph_cli.cpp
)
target_compile_options(dxapp_graph_cli_obj PRIVATE ${COMMON_COMPILE_FLAGS})
dxapp_graph_strict_flags(dxapp_graph_cli_obj)
target_compile_definitions(dxapp_graph_cli_obj PRIVATE ${COMMON_COMPILE_DEFINITIONS})
target_include_directories(dxapp_graph_cli_obj PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_ROOT}/extern/
    ${PROJECT_ROOT}/src/utility/
    ${CPP_EXAMPLE_DIR}
    ${CPP_EXAMPLE_DIR}/common/processors
    ${DXAPP_POSTPROCESS_INCLUDES}
)

# What the CLI and the dx_graph Python module share: the report schema, the
# path rule, the build sequence, the executor adapters. One object library,
# linked into every consumer, so they cannot drift. See graph_consumer.hpp.
add_library(dxapp_graph_consumer_obj OBJECT
    ${GRAPH_EXAMPLE_DIR}/graph_consumer.cpp
)
target_compile_options(dxapp_graph_consumer_obj PRIVATE ${COMMON_COMPILE_FLAGS})
dxapp_graph_strict_flags(dxapp_graph_consumer_obj)
target_compile_definitions(dxapp_graph_consumer_obj PRIVATE ${COMMON_COMPILE_DEFINITIONS})
target_include_directories(dxapp_graph_consumer_obj PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_ROOT}/extern/
    ${PROJECT_ROOT}/src/utility/
    ${CPP_EXAMPLE_DIR}
    ${CPP_EXAMPLE_DIR}/common/processors
    ${DXAPP_POSTPROCESS_INCLUDES}
)
set_target_properties(dxapp_graph_consumer_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

foreach(GRAPH_APP multi_model_graph_sync multi_model_graph_async)
    add_example(${GRAPH_APP}
        ${GRAPH_EXAMPLE_DIR}/${GRAPH_APP}.cpp "")
    # dxapp_common_obj (linked by add_example) carries the out-of-line
    # postprocessors the generated registry's factories reference.
    target_link_libraries(${GRAPH_APP}
        dxapp_graph_cli_obj dxapp_graph_consumer_obj dxapp_graph_obj
        dxapp_graph_models_obj)
    if(NOT MSVC)
        target_link_libraries(${GRAPH_APP} pthread)
    endif()
endforeach()

# =============================================================================
# dx_graph — the Python binding (optional; needs pybind11 for DXAPP_PYTHON)
# =============================================================================
# The module is built for ONE interpreter: DXAPP_PYTHON (default: the active
# virtualenv's python, else python3 on PATH). No pybind11 for it means no
# module, not a failed configure - the CLI never needs Python. A cross build
# skips it unless the target's Python is named (cmake/dxapp_platform.cmake).
option(DXAPP_BUILD_PYTHON_GRAPH "Build the dx_graph Python module" ON)
option(DXAPP_CROSS_PYTHON_GRAPH
    "Build dx_graph in a cross build too (needs -DPython_INCLUDE_DIR of the target Python)" OFF)
include(${PROJECT_ROOT}/cmake/dxapp_platform.cmake)
dxapp_python_graph_decision(DXAPP_PYTHON_GRAPH_BUILD DXAPP_PYTHON_GRAPH_WHY)
if(DXAPP_PYTHON_GRAPH_BUILD)
    if(NOT DXAPP_PYTHON)
        if(DEFINED ENV{VIRTUAL_ENV})
            set(DXAPP_PYTHON "$ENV{VIRTUAL_ENV}/bin/python")
        else()
            find_program(DXAPP_PYTHON python3)
        endif()
    endif()
    set(DXAPP_PYTHON "${DXAPP_PYTHON}" CACHE FILEPATH "Python the dx_graph module is built for")
    execute_process(COMMAND "${DXAPP_PYTHON}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE DXAPP_PYBIND11_DIR RESULT_VARIABLE DXAPP_PYBIND11_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(DXAPP_PYBIND11_RESULT EQUAL 0)
        add_subdirectory(${PROJECT_ROOT}/src/bindings/python/dx_graph
                         ${CMAKE_BINARY_DIR}/dx_graph_build)
    else()
        message(STATUS "dx_graph Python module skipped: no pybind11 for ${DXAPP_PYTHON}")
    endif()
else()
    message(STATUS "dx_graph Python module skipped: ${DXAPP_PYTHON_GRAPH_WHY}")
endif()

# =============================================================================
# Auto-Discovery: Scan category/model directories for *_sync.cpp & *_async.cpp
# =============================================================================
# Single-model categories (each subdir = one model with {model}_sync.cpp / _async.cpp)
set(CATEGORIES
    3d_object_detection
    anomaly_detection
    depth_estimation
    face_attribute
    face_detection
    face_landmark
    face_recognition
    hand_detection
    hand_landmark
    image_classification
    image_denoising
    image_matting
    image_retrieval
    instance_segmentation
    keypoint_detection
    low_light_enhancement
    mainbody_detection
    object_detection
    object_pose_estimation
    oriented_object_detection
    panoptic_driving_perception
    person_attribute
    person_reid
    pose_estimation
    semantic_segmentation
    super_resolution
    visual_place_recognition
    zero_shot_image_classification
    zero_shot_instance_segmentation
)

# One executable per variant: <task>/<family>/<variant>/<variant>_{sync,async}.cpp
function(add_variant_example VARIANT_DIR VARIANT_NAME KIND)
    set(_SRC ${VARIANT_DIR}/${VARIANT_NAME}_${KIND}.cpp)
    if(NOT EXISTS ${_SRC})
        return()
    endif()
    add_example(${VARIANT_NAME}_${KIND} ${_SRC} "")
    if(KIND STREQUAL "async" AND NOT MSVC)
        target_link_libraries(${VARIANT_NAME}_${KIND} pthread)
    endif()
endfunction()

# Sources stay under src/cpp_example. These variants are not registered and
# are not built (SCRFD500M_PPU, YOLOV5Pose_PPU).
set(DXAPP_EXCLUDED_VARIANTS SCRFD500M_PPU YOLOV5Pose_PPU)

foreach(CATEGORY ${CATEGORIES})
    file(GLOB _FAMILY_DIRS RELATIVE ${CPP_EXAMPLE_DIR}/${CATEGORY}
         ${CPP_EXAMPLE_DIR}/${CATEGORY}/*)
    foreach(FAMILY ${_FAMILY_DIRS})
        set(_FAMILY_DIR ${CPP_EXAMPLE_DIR}/${CATEGORY}/${FAMILY})
        if(NOT IS_DIRECTORY ${_FAMILY_DIR})
            continue()
        endif()

        file(GLOB _VARIANT_DIRS RELATIVE ${_FAMILY_DIR} ${_FAMILY_DIR}/*)
        foreach(VARIANT ${_VARIANT_DIRS})
            if(VARIANT IN_LIST DXAPP_EXCLUDED_VARIANTS)
                continue()
            endif()
            set(_VARIANT_DIR ${_FAMILY_DIR}/${VARIANT})
            if(NOT IS_DIRECTORY ${_VARIANT_DIR})
                continue()
            endif()
            add_variant_example(${_VARIANT_DIR} ${VARIANT} sync)
            add_variant_example(${_VARIANT_DIR} ${VARIANT} async)
        endforeach()
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
            if(NOT MSVC)
                target_link_libraries(${MODEL}_async pthread)
            endif()
        endif()
    endforeach()
endforeach()

# DOPE builds with core OpenCV only: the C++ visualizer draws the cuboid by
# connecting the predicted 2-D vertices directly (no solvePnP/projectPoints).

# =============================================================================
# Installation (Optional)
# =============================================================================
get_property(ALL_TARGETS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)

set(INSTALL_TARGETS)
foreach(_TARGET ${ALL_TARGETS})
    get_target_property(_TARGET_TYPE ${_TARGET} TYPE)
    if(NOT _TARGET_TYPE STREQUAL "UTILITY")
        list(APPEND INSTALL_TARGETS ${_TARGET})
    endif()
endforeach()

install(TARGETS ${INSTALL_TARGETS} RUNTIME DESTINATION bin)
