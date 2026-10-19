# =============================================================================
# dxapp_graph_strict_flags(<target> [SWITCH])
# =============================================================================
# The graph targets' extra strictness, spelled once per compiler family
# (SP1 U-13, SP6 U-37). Every graph target calls this instead of its own
# `if(NOT MSVC) target_compile_options(... -Werror=...)` block.
#
# GCC/Clang:
#   -Werror=switch   with SWITCH: an enumerator added without its case fails
#                    the build (the engine and the test fake switch on
#                    Shape with no default)
#   -Werror=reorder  always: an initializer list out of declaration order
#
# MSVC - NOT BUILT ON WINDOWS HERE (no MSVC on the development host; the
# spellings follow Microsoft's documentation, and tests/scripts checks only
# that they are emitted):
#   /bigobj          always: a generated registry TU holds up to 64
#                    factories and the engine/test TUs are template-heavy;
#                    past 65,279 sections MSVC stops with fatal error C1128
#   /w15038 /we5038  always: C5038 "data member A will be initialized after
#                    data member B", MSVC's -Wreorder. Off by default: /w1
#                    enables it (level 1), /we makes it an error
#   /w14062 /we4062  with SWITCH: C4062 "enumerator in switch of enum is not
#                    handled" (a switch with no default), MSVC's -Wswitch.
#                    Off by default, hence /w1 as well
include_guard(GLOBAL)

function(dxapp_graph_strict_flags target)
    cmake_parse_arguments(ARG "SWITCH" "" "" ${ARGN})
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "dxapp_graph_strict_flags(${target}): unknown argument(s) ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(MSVC)
        set(flags /bigobj /w15038 /we5038)
        if(ARG_SWITCH)
            list(APPEND flags /w14062 /we4062)
        endif()
    else()
        set(flags)
        if(ARG_SWITCH)
            list(APPEND flags -Werror=switch)
        endif()
        list(APPEND flags -Werror=reorder)
    endif()
    target_compile_options(${target} PRIVATE ${flags})
endfunction()
