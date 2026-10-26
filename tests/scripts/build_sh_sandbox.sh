#!/bin/bash
# Run a copy of build.sh inside a throwaway tree, with cmake, ninja, column,
# python3 and sudo replaced by stubs that record what they were given.
#
#   build_sh_sandbox.sh <build.sh to test> <sandbox dir> [build.sh args...]
#
# Layout it creates:
#   <sandbox>/repo     build.sh, the scripts it sources, the toolchain files,
#                      and fake build_x86_64/ build_aarch64/ bin/ lib/ include/
#                      trees, each holding a file named "marker"; the two
#                      build trees also hold release/marker
#   <sandbox>/fakebin  the stubs, first on PATH for the build.sh run
#   <sandbox>/log      cmake_argv (NUL-separated argv of the FIRST cmake call,
#                      the configure), cmake_calls (one line per cmake call),
#                      ninja (one line per ninja call), sudo (one line per
#                      call), python3, rm_denied, stdout (build.sh stdout+
#                      stderr), rc (its exit status)
#
# By default the cmake stub fails, so build.sh stops right after the
# configure call: the clean step and the configure command line are all
# that run.
#
# Environment:
#   SANDBOX_RM_FAILS=1       rm fails unless run through the sudo stub
#                            (exercises build.sh's sudo retry)
#   SANDBOX_TRACE=1          run build.sh under `bash -x` (trace in log/stdout)
#   SANDBOX_CMAKE_RC=<n>     exit status of every cmake call (default 1)
#   SANDBOX_NINJA_TARGETS=.. what `ninja -t targets` prints (default: a few
#                            phony targets and build.ninja)
#   SANDBOX_SRC_TREE=1       repo/src/cpp_example/ holds a symlink to each real
#                            category directory (the resolver reads them;
#                            nothing writes through them), and repo/run_demo.sh
#                            is a copy of the real one
#   SANDBOX_EMPTY_CATEGORY=<name>
#                            also a real, empty repo/src/cpp_example/<name>/
#   SANDBOX_INSTALL_TREE=1   a "cmake --build" call creates release/bin/ in the
#                            build dir (as an install would), and
#                            repo/src/bindings/python/dx_postprocess/ exists
#
# Never touches the real repository: every path is under <sandbox>.
set -u

src=$1
sandbox=$2
shift 2

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${here}/../.." && pwd)
repo="${sandbox}/repo"
fakebin="${sandbox}/fakebin"
log="${sandbox}/log"

mkdir -p "${repo}/scripts" "${repo}/cmake" "${fakebin}" "${log}"
cp "${src}" "${repo}/build.sh"
cp "${root}/scripts/color_env.sh" "${root}/scripts/common_util.sh" \
   "${root}/scripts/gui_env.sh" \
   "${root}/scripts/build_target_resolver.sh" "${repo}/scripts/"
cp "${root}"/cmake/toolchain.*.cmake "${repo}/cmake/"
for d in build_x86_64 build_aarch64 bin lib include; do
    mkdir -p "${repo}/${d}"
    echo keep > "${repo}/${d}/marker"
done
for d in build_x86_64 build_aarch64; do
    mkdir -p "${repo}/${d}/release"
    echo keep > "${repo}/${d}/release/marker"
done
if [ "${SANDBOX_SRC_TREE:-0}" = "1" ]; then
    mkdir -p "${repo}/src/cpp_example"
    for entry in "${root}"/src/cpp_example/*/; do
        entry=${entry%/}
        ln -s "${entry}" "${repo}/src/cpp_example/$(basename "${entry}")"
    done
    cp "${root}/run_demo.sh" "${repo}/run_demo.sh"
fi
if [ -n "${SANDBOX_EMPTY_CATEGORY:-}" ]; then
    mkdir -p "${repo}/src/cpp_example/${SANDBOX_EMPTY_CATEGORY}"
fi
if [ "${SANDBOX_INSTALL_TREE:-0}" = "1" ]; then
    mkdir -p "${repo}/src/bindings/python/dx_postprocess"
fi

cat > "${fakebin}/cmake" <<EOF
#!/bin/bash
[ -e "${log}/cmake_argv" ] || printf '%s\0' "\$@" > "${log}/cmake_argv"
echo "\$*" >> "${log}/cmake_calls"
if [ "\$1" = "--build" ] && [ "${SANDBOX_INSTALL_TREE:-0}" = "1" ]; then
    mkdir -p release/bin && echo built > release/bin/marker
fi
exit ${SANDBOX_CMAKE_RC:-1}
EOF

default_targets='yolov7_sync: phony
yolov7_async: phony
all: phony
install: phony
edit_cache: phony
rebuild_cache: phony
list_install_components: phony
build.ninja: RERUN_CMAKE'
printf '%s\n' "${SANDBOX_NINJA_TARGETS:-${default_targets}}" > "${log}/ninja_targets"

cat > "${fakebin}/ninja" <<EOF
#!/bin/bash
echo "\$*" >> "${log}/ninja"
if [ "\${1:-}" = "-t" ] && [ "\${2:-}" = "targets" ]; then
    cat "${log}/ninja_targets"
fi
exit 0
EOF

cat > "${fakebin}/column" <<'EOF'
#!/bin/bash
cat
EOF

cat > "${fakebin}/python3" <<EOF
#!/bin/bash
echo "\$*" >> "${log}/python3"
exit 1
EOF

cat > "${fakebin}/sudo" <<EOF
#!/bin/bash
echo "\$*" >> "${log}/sudo"
SUDO_STUB=1 exec "\$@"
EOF

if [ "${SANDBOX_RM_FAILS:-0}" = "1" ]; then
    cat > "${fakebin}/rm" <<EOF
#!/bin/bash
if [ -z "\${SUDO_STUB:-}" ]; then
    echo "\$*" >> "${log}/rm_denied"
    exit 1
fi
exec /bin/rm "\$@"
EOF
fi
chmod +x "${fakebin}"/*

shell=(bash)
[ "${SANDBOX_TRACE:-0}" = "1" ] && shell=(bash -x)

cd "${repo}" || exit 99
env -u VIRTUAL_ENV PATH="${fakebin}:${PATH}" "${shell[@]}" ./build.sh "$@" \
    > "${log}/stdout" 2>&1
echo $? > "${log}/rc"
exit 0
