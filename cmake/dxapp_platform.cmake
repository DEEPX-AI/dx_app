# =============================================================================
# Cross builds (SP6 U-38)
# =============================================================================
# dxapp_detect_cross_compile()
#   CROSS_COMPILE = TRUE when the target processor differs from the host's.
#   NOT CMAKE_CROSSCOMPILING: both cmake/toolchain.*.cmake set
#   CMAKE_SYSTEM_NAME, which makes CMake report CMAKE_CROSSCOMPILING=TRUE on
#   every build.sh build, the native one included.
#
# dxapp_python_graph_decision(<build_var> <why_var>)
#   Whether to build the dx_graph Python module, and why not. A cross build
#   builds it only when the target's Python is named: -DDXAPP_CROSS_PYTHON_GRAPH=ON
#   plus -DPython_INCLUDE_DIR=<target include/python3.X>. Otherwise the module
#   would be built against the host's interpreter (an x86_64 module in an
#   aarch64 install). That opt-in path is not validated here: there is no
#   target Python on the development host.
include_guard(GLOBAL)

macro(dxapp_detect_cross_compile)
    if(NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "${CMAKE_SYSTEM_PROCESSOR}")
        set(CROSS_COMPILE TRUE)
    else()
        set(CROSS_COMPILE FALSE)
    endif()
endmacro()

function(dxapp_python_graph_decision build_var why_var)
    set(build OFF)
    set(why "")
    if(NOT DXAPP_BUILD_PYTHON_GRAPH)
        set(why "DXAPP_BUILD_PYTHON_GRAPH is OFF")
    elseif(MSVC)
        set(why "no Windows build of dx_graph (MSVC)")
    elseif(CROSS_COMPILE AND NOT (DXAPP_CROSS_PYTHON_GRAPH AND Python_INCLUDE_DIR))
        set(why "cross build (host ${CMAKE_HOST_SYSTEM_PROCESSOR}, target ${CMAKE_SYSTEM_PROCESSOR}): the module would be built against the host's Python. To build it for the target, pass -DDXAPP_CROSS_PYTHON_GRAPH=ON and -DPython_INCLUDE_DIR=<the target Python's include/python3.X> (not validated here: no target Python on the development host)")
    else()
        set(build ON)
    endif()
    set(${build_var} ${build} PARENT_SCOPE)
    set(${why_var} "${why}" PARENT_SCOPE)
endfunction()
