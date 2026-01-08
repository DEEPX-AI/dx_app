cmake_minimum_required(VERSION 3.16)
project(yolov12)

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
include_directories(${PROJECT_ROOT}/src/postprocess/${PROJECT_NAME}/)

# Source files for yolov12_postprocess library
set(SOURCES
    ${PROJECT_ROOT}/src/utility/common_util.cpp
)

# Header files for yolov12_postprocess library
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

# 예제 실행 파일들 생성
set(EXAMPLE_TARGETS
    ${PROJECT_NAME}_sync
    ${PROJECT_NAME}_async
)

foreach(TARGET_NAME ${EXAMPLE_TARGETS})
    if(NOT DXAPP_WITH_SHAREDLIB)
        set(SOURCES ${PROJECT_ROOT}/src/postprocess/${PROJECT_NAME}/${PROJECT_NAME}_postprocess.cpp ${SOURCES})
    endif()
    add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME}.cpp ${SOURCES})
    if(DXAPP_WITH_SHAREDLIB)
        target_link_directories(${TARGET_NAME} PRIVATE ${PROJECT_ROOT}/lib/)
        target_link_libraries(${TARGET_NAME} PRIVATE ${PROJECT_NAME}_postprocess ${COMMON_LIBS})
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE ${COMMON_LIBS})
    endif()
    
    # pthread 추가 (async 예제용)
    if(TARGET_NAME STREQUAL "${PROJECT_NAME}_async")
        target_link_libraries(${TARGET_NAME} PRIVATE pthread)
    endif()
    
    # 공통 설정 적용
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

