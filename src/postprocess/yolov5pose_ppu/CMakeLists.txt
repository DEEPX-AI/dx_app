cmake_minimum_required(VERSION 3.16)
project(libyolov5pose_ppu)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Find required packages
find_package(PkgConfig REQUIRED)

# Find dxrt library
find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
set(DXRT_LIB dxrt)
if(CROSS_COMPILE)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib)
endif()
if(NOT DXRT_LIB)
    message(FATAL_ERROR "dxrt library not found. Please ensure it is installed and accessible.")
endif()

if(NOT PROJECT_ROOT)
    # Set the project root directory (parent of parent directory)
    get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
endif()

# Include directories
include_directories(${PROJECT_ROOT}/src/utility/)

# Source files for yolov5pose_ppu_postprocess library
set(SOURCES
    ${PROJECT_ROOT}/src/utility/common_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov5pose_ppu_postprocess.cpp
)

# Header files for yolov5pose_ppu_postprocess library
set(HEADERS
    ${PROJECT_ROOT}/src/utility/common_util.hpp
    ${PROJECT_ROOT}/src/utility/common_util_inline.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov5pose_ppu_postprocess.h
)

# Common libraries for linking
set(COMMON_LIBS ${DXRT_LIB})

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

# yolov5pose_ppu_postprocess 라이브러리 생성 (공유 라이브러리, .so 파일)
if(SOURCES)
    add_library(yolov5pose_ppu_postprocess SHARED ${SOURCES} ${HEADERS})
    target_link_libraries(yolov5pose_ppu_postprocess PRIVATE ${COMMON_LIBS})
    
    # 라이브러리 속성 설정
    set_target_properties(yolov5pose_ppu_postprocess PROPERTIES
        OUTPUT_NAME "dxapp_yolov5pose_ppu_postprocess"
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_ROOT}/lib
    )
    
    # Copy header files to build directory during build
    add_custom_command(TARGET yolov5pose_ppu_postprocess POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_ROOT}/include
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HEADERS} ${PROJECT_ROOT}/include
        COMMENT "Copying header files to build directory"
    )
    
    # Compiler flags and definitions
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(yolov5pose_ppu_postprocess PRIVATE ${COMMON_COMPILE_FLAGS})
    endif()
    target_compile_definitions(yolov5pose_ppu_postprocess PRIVATE ${COMMON_COMPILE_DEFINITIONS})
else()
    message(WARNING "No source files found for yolov5pose_ppu_postprocess. The library will not be built.")
endif()

# Install targets (optional - only if make install is used)
install(TARGETS yolov5pose_ppu_postprocess
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install header files (optional - only if make install is used)
install(FILES ${HEADERS}
    DESTINATION include
)


