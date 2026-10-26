# =============================================================================
# dxapp_find_codegen_python()
# =============================================================================
# The configure step generates the graph model registry with
# scripts/gen_model_registry.py, so it needs a Python 3 interpreter - often
# missing on Windows (SP6 U-37). Say so, and how to fix it, instead of
# FindPython3's generic "Could NOT find Python3". A macro, so
# Python3_EXECUTABLE reaches the caller's scope.
include_guard(GLOBAL)

macro(dxapp_find_codegen_python)
    find_package(Python3 COMPONENTS Interpreter)
    if(NOT Python3_Interpreter_FOUND)
        if(Python3_EXECUTABLE)
            set(_dxapp_python_hint
                " (-DPython3_EXECUTABLE=${Python3_EXECUTABLE} is not a usable Python 3)")
        else()
            set(_dxapp_python_hint "")
        endif()
        message(FATAL_ERROR
            "No Python 3 interpreter for the graph model registry${_dxapp_python_hint}.\n"
            "The configure step runs scripts/gen_model_registry.py to generate it. "
            "Install Python 3 (Linux: apt install python3; Windows: "
            "https://www.python.org/downloads/ - the Microsoft Store stub does not work), "
            "or name one with -DPython3_EXECUTABLE=<path to python3>.")
    endif()
endmacro()
