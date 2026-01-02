cmake_minimum_required(VERSION 3.16)
project(efficientnet_async)

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
find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
set(DXRT_LIB dxrt)
if(CROSS_COMPILE)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib)
endif()
if(NOT DXRT_LIB)
    message(FATAL_ERROR "dxrt library not found. Please ensure it is installed and accessible.")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_ROOT}/extern/)
include_directories(${PROJECT_ROOT}/src/utility/)  # Add utility directory

# Source files
set(SOURCES
    ${PROJECT_ROOT}/src/utility/common_util.cpp  # Add utility implementation file
)

# Header files
set(HEADERS
    # Utility headers for common functionality
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

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpp ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${COMMON_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_ROOT}/bin)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE ${COMMON_COMPILE_FLAGS})
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMMON_COMPILE_DEFINITIONS})

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

