# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(MSVC)
    # Verify that pybind11 is installed; install it using pip if it is missing
    execute_process(
        COMMAND pybind11-config --cmakedir
        OUTPUT_VARIABLE pybind11_CMAKE_DIR
        RESULT_VARIABLE pybind11_check_result
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

target_include_directories(dx_postprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(MSVC)
    target_link_libraries(dx_postprocess PRIVATE
        "$<$<CONFIG:Debug>:${Python_LIBRARY_RELEASE}>"
        "$<$<CONFIG:Release>:${Python_LIBRARY_RELEASE}>"
    )
else()
    execute_process(
        COMMAND cp "${CMAKE_SOURCE_DIR}/release.ver" "${PYTHON_SITE_PACKAGES}/test.tmp"
        RESULT_VARIABLE COPY_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT COPY_RESULT STREQUAL "0")
        set(CMAKE_INSTALL_PREFIX "~/.local")
        set(PYTHON_VERSION ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
        set(PYTHON_SITE_PACKAGES ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages)
    endif()
endif()

message(STATUS "Python Site-Packages Directory: ${PYTHON_SITE_PACKAGES}")

install(TARGETS dx_postprocess DESTINATION ${PYTHON_SITE_PACKAGES})