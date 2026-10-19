#!/usr/bin/env bash
# aarch64 compile check (SP6 U-38): the graph engine and CLI sources compile
# for aarch64 with the cross compiler, as C++14, with the graph targets'
# -Werror flags. -fsyntax-only: there is no aarch64 dxrt or OpenCV to link
# against on the development host, nor on CI. Nothing here RUNS on aarch64;
# concurrency was validated with x86_64 dxrt only.
#
#   bash scripts/check_cross_compile.sh [--root <dir>] [--with-registry]
#
#   --root <dir>      check the sources of another tree (a copy under test)
#   --with-registry   also the generated registry TUs (all factories, about
#                     100 s), generated into a temp dir
#
# Environment: DXAPP_CROSS_CXX (default aarch64-linux-gnu-g++),
# DXRT_INCLUDE_DIR (default /usr/local/include; holds dxrt/),
# OPENCV_INCLUDE_DIR (default /usr/include/opencv4). Headers only:
#   - the dxrt headers are checked for x86-only code first - none outside
#     dxrt/extern/, whose rapidjson #ifs are portable; gen.h is build
#     configuration (USE_ORT, ...), printed, not architecture;
#   - the OpenCV public headers are architecture-independent (cvconfig.h,
#     the per-arch generated header, lives in the multiarch directory and
#     is not included by them).
# The postprocess include roots mirror DXAPP_POSTPROCESS_INCLUDES in
# src/cpp_example/CMakeLists.txt (factories include those headers by bare
# name, e.g. vitpose_postprocessor.hpp: "vitpose_postprocess.h");
# tests/scripts/test_cross_compile.py keeps the two lists in step.
# Flags per source follow the real targets: common/graph/*.cpp as
# dxapp_graph_obj (-Werror=switch -Werror=reorder); graph_cli.cpp,
# graph_consumer.cpp, static_model_registry.cpp and the registry TUs with
# -Werror=reorder; the other multi_model_graph/*.cpp (the two mains) with
# the base flags only.
#
# Exit: 0 OK, 1 a check failed, 2 a prerequisite is missing.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
with_registry=0
while [ $# -gt 0 ]; do
    case "$1" in
        --root)
            if [ -z "${2:-}" ] || [ ! -d "${2}/src/cpp_example/common/graph" ]; then
                echo "check_cross_compile: --root needs a tree holding src/cpp_example/common/graph" >&2
                exit 2
            fi
            ROOT="$(cd "$2" && pwd -P)"
            shift 2 ;;
        --with-registry) with_registry=1; shift ;;
        *) echo "check_cross_compile: unknown argument: $1" >&2; exit 2 ;;
    esac
done
CXX_CROSS="${DXAPP_CROSS_CXX:-aarch64-linux-gnu-g++}"
DXRT_INC="${DXRT_INCLUDE_DIR:-/usr/local/include}"
OPENCV_INC="${OPENCV_INCLUDE_DIR:-/usr/include/opencv4}"
CE="$ROOT/src/cpp_example"

command -v "$CXX_CROSS" >/dev/null 2>&1 || { echo "check_cross_compile: $CXX_CROSS not found" >&2; exit 2; }
[ -f "$DXRT_INC/dxrt/dxrt_api.h" ] || { echo "check_cross_compile: no dxrt headers at $DXRT_INC/dxrt" >&2; exit 2; }
[ -f "$OPENCV_INC/opencv2/core.hpp" ] || { echo "check_cross_compile: no OpenCV headers at $OPENCV_INC" >&2; exit 2; }

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
# Only dxrt/ from DXRT_INC: the rest of /usr/local/include (onnxruntime, ...)
# stays off the include path.
mkdir -p "$W/inc"
ln -s "$DXRT_INC/dxrt" "$W/inc/dxrt"

# 1. No x86-only code in the dxrt headers.
arch_hits=$(grep -rlE '__x86_64__|__i386__|_M_X64|_M_IX86|mmintrin\.h|__SSE|__AVX' "$DXRT_INC/dxrt" 2>/dev/null \
            | grep -v '/extern/' || true)
if [ -n "$arch_hits" ]; then
    echo "FAILED: x86-specific code in the dxrt headers (an aarch64 dxrt may differ):"
    echo "$arch_hits"
    exit 1
fi
config=$(grep -hoE '^#define USE_[A-Z_]+' "$DXRT_INC/dxrt/gen.h" 2>/dev/null | awk '{print $2}' | tr '\n' ' ')
echo "dxrt headers: no x86-only code outside extern/ (build configuration in gen.h: ${config:-none})"

# 2. Every graph engine and CLI TU, with its target's flags.
BASE=(-std=gnu++14 -W -Wall -Wextra -Werror=c++17-extensions -pthread -fsyntax-only)
INC=(-I"$CE" -I"$CE/common/processors" -I"$ROOT/extern" -I"$ROOT/src/utility"
     -I"$ROOT/src/postprocess/sfa3d" -I"$ROOT/src/postprocess/superpoint"
     -I"$ROOT/src/postprocess/dope" -I"$ROOT/src/postprocess/yolopv2"
     -I"$ROOT/src/postprocess/vitpose"
     -isystem "$OPENCV_INC" -I"$W/inc")
DEFS=(-DPROJECT_ROOT_DIR="\"$ROOT\"")
status=0
count=0

check() {  # check [extra flags...] -- files...
    local extra=() f
    while [ "$1" != "--" ]; do extra+=("$1"); shift; done
    shift
    for f in "$@"; do
        [ -f "$f" ] || continue
        count=$((count + 1))
        if ! "$CXX_CROSS" "${BASE[@]}" "${extra[@]}" "${INC[@]}" "${DEFS[@]}" "$f" > "$W/out.log" 2>&1; then
            cat "$W/out.log"
            # Relative to src/cpp_example, or to the temp dir of the registry
            # TUs (the EXIT trap deletes it).
            rel=${f#"$CE"/}
            rel=${rel#"$W"/}
            echo "FAILED: $rel does not compile for aarch64"
            status=1
        fi
    done
}

check -Werror=switch -Werror=reorder -- "$CE"/common/graph/*.cpp
check -Werror=reorder -- "$CE/multi_model_graph/graph_cli.cpp" "$CE/multi_model_graph/graph_consumer.cpp" \
    "$CE/common/registry/static_model_registry.cpp"
mains=()
for f in "$CE"/multi_model_graph/*.cpp; do
    case "$(basename "$f")" in graph_cli.cpp|graph_consumer.cpp) ;; *) mains+=("$f") ;; esac
done
check -- "${mains[@]}"
if [ "$with_registry" = 1 ]; then
    python3 "$ROOT/scripts/gen_model_registry.py" --out-dir "$W/generated" --docs "$W/graph_models.md" \
        --manifest "$ROOT/scripts/modelzoo_manifest.json" > /dev/null \
        || { echo "FAILED: gen_model_registry.py"; exit 1; }
    check -Werror=reorder -- "$W"/generated/*.cpp
fi

if [ "$status" -ne 0 ]; then
    exit 1
fi
echo "check_cross_compile OK: $count sources compile for aarch64 ($CXX_CROSS, -fsyntax-only)"
