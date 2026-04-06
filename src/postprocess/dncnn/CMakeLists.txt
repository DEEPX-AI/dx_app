cmake_minimum_required(VERSION 3.14)
project(libdncnn)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Find required packages
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

if(NOT PROJECT_ROOT)
    get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
endif()

# Source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/dncnn_postprocess.cpp
)

# Header files
set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/dncnn_postprocess.h
)

# Common libraries for linking
set(COMMON_LIBS ${DXRT_LIB})

if(NOT STD_FS_NO_LIB_NEEDED)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND COMMON_LIBS stdc++fs)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND COMMON_LIBS c++fs)
    endif()
endif()

set(COMMON_COMPILE_FLAGS -Wall -Wextra -O3)
set(COMMON_COMPILE_DEFINITIONS PROJECT_ROOT_DIR="${PROJECT_ROOT}")

if(SOURCES)
    add_library(dncnn_postprocess SHARED ${SOURCES} ${HEADERS})
    target_link_libraries(dncnn_postprocess PRIVATE ${COMMON_LIBS})

    set_target_properties(dncnn_postprocess PROPERTIES
        OUTPUT_NAME "dxapp_dncnn_postprocess"
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_ROOT}/lib
    )
    add_custom_command(TARGET dncnn_postprocess POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_ROOT}/include
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HEADERS} ${PROJECT_ROOT}/include
        COMMENT "Copying header files to build directory"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(dncnn_postprocess PRIVATE ${COMMON_COMPILE_FLAGS})
    endif()
    target_compile_definitions(dncnn_postprocess PRIVATE ${COMMON_COMPILE_DEFINITIONS})
else()
    message(WARNING "No source files found for dncnn_postprocess. The library will not be built.")
endif()

install(TARGETS dncnn_postprocess
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
    DESTINATION include
)
