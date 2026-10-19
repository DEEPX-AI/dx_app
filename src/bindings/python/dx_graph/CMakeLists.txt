# =============================================================================
# dx_graph — the multi-model graph engine as a Python module
# =============================================================================
# Added by src/cpp_example/CMakeLists.txt, never on its own: it links the
# object libraries defined there (dxapp_graph_consumer_obj, dxapp_graph_obj,
# dxapp_graph_models_obj, dxapp_common_obj), which is what keeps the module
# and the multi_model_graph CLI on the same code. DXAPP_PYTHON and
# DXAPP_PYBIND11_DIR come from there too.
#
# Output: ${CMAKE_BINARY_DIR}/python/dx_graph/ - a complete package
# (_dx_graph.<ext> plus the .py files). Installed to
# ${CMAKE_INSTALL_PREFIX}/bin/python/dx_graph, next to the CLI binaries in
# bin/, so `./build.sh --all` (install, then copy the prefix into the
# repository) ships it as bin/python/dx_graph. Put bin/python on PYTHONPATH.
# The example scripts (examples/) go beside the package, not into it:
# ${CMAKE_BINARY_DIR}/python/dx_graph_examples/, installed as
# bin/python/dx_graph_examples/.
set(Python_EXECUTABLE "${DXAPP_PYTHON}")
set(PYBIND11_FINDPYTHON ON)
# Found here, explicitly, before pybind11: pybind11 reuses an already-found
# Python3 package in preference to finding its own, and the parent directory
# has found Python3 - the interpreter the registry codegen runs on, which is
# not necessarily DXAPP_PYTHON and has no Development component.
# Not REQUIRED: an interpreter without its development headers means no
# module, as no pybind11 does - the CLI never needs Python.
find_package(Python COMPONENTS Interpreter Development.Module)
if(NOT Python_FOUND)
    message(STATUS "dx_graph Python module skipped: no Python development headers (Development.Module) for ${DXAPP_PYTHON}")
    return()
endif()
find_package(pybind11 CONFIG REQUIRED PATHS "${DXAPP_PYBIND11_DIR}" NO_DEFAULT_PATH)

set(DX_GRAPH_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/dx_graph)
pybind11_add_module(_dx_graph ${CMAKE_CURRENT_SOURCE_DIR}/dx_graph_pybinding.cpp)
target_compile_options(_dx_graph PRIVATE ${COMMON_COMPILE_FLAGS})
dxapp_graph_strict_flags(_dx_graph)
target_compile_definitions(_dx_graph PRIVATE ${COMMON_COMPILE_DEFINITIONS}
    DX_GRAPH_PYTHON_VERSION="${Python_VERSION}")
target_include_directories(_dx_graph PRIVATE ${OpenCV_INCLUDE_DIRS} ${PROJECT_ROOT}/extern/
    ${PROJECT_ROOT}/src/utility/ ${CPP_EXAMPLE_DIR} ${CPP_EXAMPLE_DIR}/common/processors
    ${DXAPP_POSTPROCESS_INCLUDES})
target_link_libraries(_dx_graph PRIVATE dxapp_graph_consumer_obj dxapp_graph_obj
    dxapp_graph_models_obj dxapp_common_obj ${COMMON_LIBS} pthread)
set_target_properties(_dx_graph PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${DX_GRAPH_PACKAGE_DIR})

# The package's pure-Python files. Copied by a build rule rather than
# configure_file, so an edit reaches the package on the next `ninja`
# without re-running the configure step (which regenerates the registry).
set(DX_GRAPH_PY_FILES __init__.py)
set(DX_GRAPH_PY_OUTPUTS)
foreach(PY_FILE ${DX_GRAPH_PY_FILES})
    add_custom_command(
        OUTPUT ${DX_GRAPH_PACKAGE_DIR}/${PY_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DX_GRAPH_PACKAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/dx_graph/${PY_FILE}
                ${DX_GRAPH_PACKAGE_DIR}/${PY_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dx_graph/${PY_FILE}
        COMMENT "dx_graph: ${PY_FILE}")
    list(APPEND DX_GRAPH_PY_OUTPUTS ${DX_GRAPH_PACKAGE_DIR}/${PY_FILE})
endforeach()
set(DX_GRAPH_EXAMPLES_DIR ${CMAKE_BINARY_DIR}/python/dx_graph_examples)
set(DX_GRAPH_EXAMPLE_FILES run_graph.py)
foreach(EXAMPLE_FILE ${DX_GRAPH_EXAMPLE_FILES})
    add_custom_command(
        OUTPUT ${DX_GRAPH_EXAMPLES_DIR}/${EXAMPLE_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DX_GRAPH_EXAMPLES_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/${EXAMPLE_FILE}
                ${DX_GRAPH_EXAMPLES_DIR}/${EXAMPLE_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${EXAMPLE_FILE}
        COMMENT "dx_graph example: ${EXAMPLE_FILE}")
    list(APPEND DX_GRAPH_PY_OUTPUTS ${DX_GRAPH_EXAMPLES_DIR}/${EXAMPLE_FILE})
endforeach()
add_custom_target(dx_graph_python_files ALL DEPENDS ${DX_GRAPH_PY_OUTPUTS})
add_dependencies(_dx_graph dx_graph_python_files)

# The same package, installed where the CLI binaries are installed (bin/).
install(TARGETS _dx_graph LIBRARY DESTINATION bin/python/dx_graph)
foreach(PY_FILE ${DX_GRAPH_PY_FILES})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/dx_graph/${PY_FILE}
            DESTINATION bin/python/dx_graph)
endforeach()
foreach(EXAMPLE_FILE ${DX_GRAPH_EXAMPLE_FILES})
    install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${EXAMPLE_FILE}
            DESTINATION bin/python/dx_graph_examples)
endforeach()

message(STATUS "dx_graph Python module: ${DX_GRAPH_PACKAGE_DIR} (Python ${Python_VERSION}, pybind11 ${pybind11_VERSION})")
