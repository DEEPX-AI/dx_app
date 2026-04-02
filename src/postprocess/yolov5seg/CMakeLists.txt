cmake_minimum_required(VERSION 3.14)
project(libyolov5seg)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(PkgConfig REQUIRED)

if(CROSS_COMPILE OR MSVC)
find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
include_directories(${DXRT_INSTALLED_DIR}/include)
link_directories(${DXRT_INSTALLED_DIR}/lib)
else()
find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
set(DXRT_LIB dxrt)
endif()
if(NOT DXRT_LIB)
    message(FATAL_ERROR "dxrt library not found.")
endif()

if(NOT PROJECT_ROOT)
    get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
endif()

include_directories(${PROJECT_ROOT}/src/utility/)

set(SOURCES
    ${PROJECT_ROOT}/src/utility/common_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov5seg_postprocess.cpp
)

set(HEADERS
    ${PROJECT_ROOT}/src/utility/common_util.hpp
    ${PROJECT_ROOT}/src/utility/common_util_inline.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov5seg_postprocess.h
)

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
    add_library(yolov5seg_postprocess SHARED ${SOURCES} ${HEADERS})
    target_link_libraries(yolov5seg_postprocess PRIVATE ${COMMON_LIBS})

    set_target_properties(yolov5seg_postprocess PROPERTIES
        OUTPUT_NAME "dxapp_yolov5seg_postprocess"
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_ROOT}/lib
    )
    add_custom_command(TARGET yolov5seg_postprocess POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_ROOT}/include
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HEADERS} ${PROJECT_ROOT}/include
        COMMENT "Copying header files to build directory"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(yolov5seg_postprocess PRIVATE ${COMMON_COMPILE_FLAGS})
    endif()
    target_compile_definitions(yolov5seg_postprocess PRIVATE ${COMMON_COMPILE_DEFINITIONS})
endif()

install(TARGETS yolov5seg_postprocess
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${HEADERS} DESTINATION include)
