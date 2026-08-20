# Stage every runtime DLL from SRC_DIR into DST_DIR.
#
# Invoked at build time as:
#   cmake -DSRC_DIR=<vcpkg bin dir> -DDST_DIR=<output dir> -P stage_vcpkg_dlls.cmake
#
# This replaces vcpkg's per-target `vcpkg z-applocal` POST_BUILD step
# (VCPKG_APPLOCAL_DEPS). That step races when many executables share a single
# output directory under a parallel build; see src/cpp_example/CMakeLists.txt.

if(NOT DEFINED SRC_DIR OR NOT DEFINED DST_DIR)
    message(FATAL_ERROR "stage_vcpkg_dlls.cmake requires -DSRC_DIR=... -DDST_DIR=...")
endif()

if(NOT IS_DIRECTORY "${SRC_DIR}")
    message(STATUS "stage_vcpkg_dlls: no such directory, nothing to stage: ${SRC_DIR}")
    return()
endif()

file(GLOB _dlls "${SRC_DIR}/*.dll")
if(NOT _dlls)
    message(STATUS "stage_vcpkg_dlls: no DLLs found in ${SRC_DIR}")
    return()
endif()

file(MAKE_DIRECTORY "${DST_DIR}")

# file(COPY) skips files already present with a matching timestamp and size,
# so incremental rebuilds cost close to nothing.
file(COPY ${_dlls} DESTINATION "${DST_DIR}")

list(LENGTH _dlls _count)
message(STATUS "stage_vcpkg_dlls: staged ${_count} DLL(s) into ${DST_DIR}")
