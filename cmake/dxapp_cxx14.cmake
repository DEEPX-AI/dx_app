# =============================================================================
# dxapp_cxx14_conformance_flags(<out_var>)
# =============================================================================
# C++14 conformance for GCC/Clang: a C++17 extension (a structured binding,
# an inline variable, ...) compiled under -std=gnu++14 is only a warning;
# -Werror=c++17-extensions makes it fail the build. Shared by the main build
# (CMakeLists.txt) and the separate dx_postprocess pip project
# (src/bindings/python/dx_postprocess/CMakeLists.txt), which the main
# build's add_compile_options never reached (U-39: dope_postprocess.cpp once
# held a structured binding only that project compiled).
#
# MSVC: nothing - it is built as C++17 (see "MSVC stays C++17" in
# CMakeLists.txt) and has no such diagnostic name.
include_guard(GLOBAL)
include(CheckCXXCompilerFlag)

function(dxapp_cxx14_conformance_flags out_var)
    set(flags)
    if(NOT MSVC)
        check_cxx_compiler_flag("-Werror=c++17-extensions" DXAPP_HAS_CXX17_EXTENSIONS_WARNING)
        if(DXAPP_HAS_CXX17_EXTENSIONS_WARNING)
            list(APPEND flags -Werror=c++17-extensions)
        endif()
    endif()
    set(${out_var} ${flags} PARENT_SCOPE)
endfunction()
