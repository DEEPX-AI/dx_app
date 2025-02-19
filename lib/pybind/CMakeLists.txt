# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# pybind11 설치 여부 확인 및 필요 시 pip로 설치
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_CMAKE_DIR
    RESULT_VARIABLE pybind11_check_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT pybind11_check_result EQUAL 0)
    message(STATUS "pybind11 not found, installing via pip...")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pip install pybind11
        RESULT_VARIABLE pip_install_result
    )
    if(NOT pip_install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install pybind11 via pip.")
    endif()

    # 다시 pybind11 CMake 디렉토리 찾기
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# Append the pybind11 CMake directory to CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${pybind11_CMAKE_DIR})

find_package(pybind11 REQUIRED)

set(SRC_FILES
    yolo_post_processing.cpp
    yolo_post_processing_pybinding.cpp
)

pybind11_add_module(dx_postprocess ${SRC_FILES})

# 헤더 파일 찾기
target_include_directories(dx_postprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 설치 경로 설정
set(CMAKE_INSTALL_PREFIX "~/.local")
set(PYTHON_VERSION ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
set(PYTHON_SITE_PACKAGES ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages)

# 설치
install(TARGETS dx_postprocess DESTINATION ${PYTHON_SITE_PACKAGES})