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

    if(MSVC)
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
    else()
        execute_process(
            COMMAND ${Python_EXECUTABLE} -m pipx run pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
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
if(MSVC)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_EXECUTABLE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(PYTHON_SITE_PACKAGES ${PYTHON_EXECUTABLE_PATH}\\Lib\\site-packages)
    message(STATUS "Find Python Site-Packages Directory: ${PYTHON_SITE_PACKAGES}")
else()
    execute_process(
        COMMAND id -u
        OUTPUT_VARIABLE USER_ID
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(USER_ID EQUAL "0")
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
        set(CMAKE_INSTALL_PREFIX "~/.local")
        set(PYTHON_VERSION ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
        set(PYTHON_SITE_PACKAGES ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages)
    endif()
endif()

# 설치
install(TARGETS dx_postprocess DESTINATION ${PYTHON_SITE_PACKAGES})