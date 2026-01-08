cmake_minimum_required(VERSION 3.16)
project(rtsp_examples)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT PROJECT_ROOT)
    get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../.. ABSOLUTE)
endif()

# Find required packages
find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR})
find_package(PkgConfig REQUIRED)

# Find FFmpeg libraries (optional)
find_package(PkgConfig)
# Find GStreamer libraries (optional)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(GSTREAMER QUIET
        gstreamer-1.0
        gstreamer-app-1.0
        gstreamer-video-1.0
    )
endif()

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

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_ROOT}/extern/)
include_directories(${PROJECT_ROOT}/src/utility/)

set(SOURCES
    ${PROJECT_ROOT}/src/utility/common_util.cpp
)

set(HEADERS
    ${PROJECT_ROOT}/src/utility/common_util.hpp
    ${PROJECT_ROOT}/src/utility/common_util_inline.hpp
)

# Common libraries for linking
set(COMMON_LIBS ${OpenCV_LIBS} ${DXRT_LIB})

# Platform-specific filesystem library linking for C++17
if(NOT STD_FS_NO_LIB_NEEDED)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # GCC < 9.0 requires explicit linking to stdc++fs
        list(APPEND COMMON_LIBS stdc++fs)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Clang < 9.0 requires explicit linking to c++fs
        list(APPEND COMMON_LIBS c++fs)
    endif()
endif()
# MSVC (Windows) and newer GCC/Clang don't need explicit filesystem library linking

# Common compiler flags
set(COMMON_COMPILE_FLAGS -Wall -Wextra -O3 )

# Common compile definitions
set(COMMON_COMPILE_DEFINITIONS PROJECT_ROOT_DIR="${PROJECT_ROOT}")

set(EXAMPLE_TARGETS
    rtsp_opencv
)
if(NOT CROSS_COMPILE AND NOT MSVC)
    list(APPEND EXAMPLE_TARGETS rtsp_gstreamer)
endif()

foreach(TARGET_NAME ${EXAMPLE_TARGETS})
    add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME}.cpp ${SOURCES})
    
    # Base libraries
    target_link_libraries(${TARGET_NAME} PRIVATE ${COMMON_LIBS} pthread)
    
    # GStreamer libraries for rtsp_gstreamer target
    if(TARGET_NAME STREQUAL "rtsp_gstreamer")
        if(GSTREAMER_FOUND)
            target_include_directories(${TARGET_NAME} PRIVATE ${GSTREAMER_INCLUDE_DIRS})
            target_compile_options(${TARGET_NAME} PRIVATE ${GSTREAMER_CFLAGS_OTHER})
            target_link_libraries(${TARGET_NAME} PRIVATE ${GSTREAMER_LIBRARIES})
            target_link_directories(${TARGET_NAME} PRIVATE ${GSTREAMER_LIBRARY_DIRS})
            message(STATUS "GStreamer libraries found for ${TARGET_NAME}")
        else()
            message(WARNING "GStreamer libraries not found. Install with: sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev")
        endif()
    endif()
    
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_ROOT}/bin
    )
    
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_FLAGS})
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE ${COMMON_COMPILE_DEFINITIONS})

    install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION bin
    )
endforeach()

