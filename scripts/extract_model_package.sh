#!/bin/bash
# =============================================================================
# extract_model_package.sh - Prepare model directories for standalone deployment (C++/Python)
# =============================================================================
# Usage:
#   ./extract_model_package.sh <model_dir|all> [--lang cpp|py|both] [--output-dir <path>] [--no-prune] [--clean]
#
# C++ mode:  Copy common/, utility/, extern/, generate CMakeLists.txt
# Python mode: Copy common/
#
# Options:
#   --output-dir <path>  Export standalone package to a separate directory
#                        instead of modifying the source tree in-place.
#                        Creates <path>/<lang>/<category>/<model>/
#   --no-prune           Copy the whole common/ tree instead of only the files
#                        the model depends on. Pruning is ON by default.
#   --prune              Explicitly request pruning (the default). Unlike the
#                        default, a package with no sources is a hard error.
#
# Examples:
#   ./extract_model_package.sh object_detection/yolov7
#   ./extract_model_package.sh depth_estimation/depthanythingv2/depthanythingv2-vitb_224x224 --output-dir /tmp/x --lang py
#   ./extract_model_package.sh all
#   ./extract_model_package.sh all --lang cpp
#   ./extract_model_package.sh all --lang py
#   ./extract_model_package.sh object_detection/yolov7 --clean
#   ./extract_model_package.sh all --clean --lang both
#   ./extract_model_package.sh object_detection/yolov7 --output-dir outputs/
#   ./extract_model_package.sh all --output-dir /tmp/standalone --lang cpp
#   ./extract_model_package.sh object_detection/yolov7 --output-dir outputs/ --no-prune

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DX_APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_SRC_DIR="$DX_APP_ROOT/src/cpp_example"
PY_SRC_DIR="$DX_APP_ROOT/src/python_example"
CPP_COMMON_SRC="$CPP_SRC_DIR/common"
PY_COMMON_SRC="$PY_SRC_DIR/common"
UTILITY_SRC="$DX_APP_ROOT/src/utility"
EXTERN_SRC="$DX_APP_ROOT/extern"

# Colors
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
NC='\033[0m'

# =========================================================================
# Usage
# =========================================================================
usage() {
    echo "Usage: $0 <model_dir|all> [--lang cpp|py|both] [--output-dir <path>] [--no-prune] [--clean]"
    echo ""
    echo "Arguments:"
    echo "  model_dir           Family (task/family) or one model (task/family/variant)"
    echo "  all                 Prepare all model directories"
    echo "  --lang <mode>       Language: cpp, py, or both (default: both)"
    echo "  --output-dir <path> Export to a separate directory (default: in-place)"
    echo "  --no-prune          Copy all of common/ (default: prune to what is used)"
    echo "  --clean             Remove standalone files instead of creating them"
    echo ""
    echo "Examples:"
    echo "  $0 object_detection/yolov7"
    echo "  $0 all --lang cpp"
    echo "  $0 object_detection/yolov8 --lang py"
    echo "  $0 all --clean"
    echo "  $0 all --output-dir outputs/"
    echo "  $0 object_detection/yolov7 --output-dir /tmp/deploy --lang cpp"
    echo "  $0 object_detection/yolov7 --output-dir /tmp/deploy --no-prune"
    exit 1
}

# =========================================================================
# Parse arguments
# =========================================================================
if [ $# -eq 0 ]; then usage; fi

CLEAN_MODE=false
MODEL_ARG=""
LANG_MODE="both"
OUTPUT_DIR=""
# Pruning is the default: a standalone package should carry what it uses, not the
# whole framework. --no-prune restores the historical full copy.
PRUNE_MODE=true
PRUNE_EXPLICIT=false

while [ $# -gt 0 ]; do
    case "$1" in
        --clean)      CLEAN_MODE=true;  shift ;;
        --no-prune)   PRUNE_MODE=false; shift ;;
        --prune)      PRUNE_MODE=true;  PRUNE_EXPLICIT=true; shift ;;
        --lang)       LANG_MODE="$2";   shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        -h|--help)    usage ;;
        *)            MODEL_ARG="$1";   shift ;;
    esac
done

if [ -z "$MODEL_ARG" ]; then usage; fi

if [ "$PRUNE_EXPLICIT" = true ] && [ "$CLEAN_MODE" = true ]; then
    echo -e "${RED}[DXAPP] [ERROR]${NC} --prune cannot be combined with --clean"
    exit 1
fi

PRUNE_SCRIPT="$SCRIPT_DIR/prune_common.py"
if [ "$PRUNE_MODE" = true ] && [ ! -f "$PRUNE_SCRIPT" ]; then
    echo -e "${RED}[DXAPP] [ERROR]${NC} --prune requires $PRUNE_SCRIPT"
    exit 1
fi

# Drop common/ files the package does not reach. Failure aborts the extraction
# rather than leaving a package with a half-pruned common/.
run_prune() {
    local target_dir="$1"
    local lang="$2"
    [ "$PRUNE_MODE" = true ] || return 0
    # When the user did not ask for pruning, a package we cannot analyse (no
    # sources) must not fail the extraction — fall back to the full copy.
    local lenient=""
    [ "$PRUNE_EXPLICIT" = true ] || lenient="--allow-no-roots"
    python3 "$PRUNE_SCRIPT" --lang "$lang" "$target_dir" $lenient
}

# Resolve output-dir to absolute path
if [ -n "$OUTPUT_DIR" ]; then
    if [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="$(cd "$DX_APP_ROOT" && pwd)/$OUTPUT_DIR"
    fi
    if [ "$CLEAN_MODE" = true ]; then
        echo -e "${YELLOW}[DXAPP] [WARN]${NC} --clean is ignored with --output-dir"
        CLEAN_MODE=false
    fi
fi

case "$LANG_MODE" in
    cpp)  LANGS=("cpp") ;;
    py)   LANGS=("py") ;;
    both) LANGS=("cpp" "py") ;;
    *)    echo "Invalid --lang: $LANG_MODE (cpp|py|both)"; exit 1 ;;
esac

# =========================================================================
# C++ - Remove standalone files
# =========================================================================
clean_model_cpp() {
    local model_dir="$1"
    local target_dir="$CPP_SRC_DIR/$model_dir"

    if [ ! -d "$target_dir" ]; then
        echo -e "${RED}[DXAPP] [ERROR]${NC} C++ directory not found: $model_dir"
        return 1
    fi

    rm -rf "$target_dir/common" \
           "$target_dir/postprocess" \
           "$target_dir/utility" \
           "$target_dir/extern" \
           "$target_dir/CMakeLists.txt"

    echo -e "${GREEN}[CLEAN]${NC} C++ $model_dir"
}

# =========================================================================
# C++ - Generate standalone CMakeLists.txt
# =========================================================================
generate_cmake() {
    local target_dir="$1"
    local model_name="$2"

    # Discover source files
    local sync_sources=()
    local async_sources=()
    while IFS= read -r -d '' f; do
        sync_sources+=("${f#"$target_dir"/}")
    done < <(find "$target_dir" -name "*_sync.cpp" ! -path "*/common/*" -print0 2>/dev/null)
    while IFS= read -r -d '' f; do
        async_sources+=("${f#"$target_dir"/}")
    done < <(find "$target_dir" -name "*_async.cpp" ! -path "*/common/*" -print0 2>/dev/null)

    # pthread is linked into every target via Threads::Threads in COMMON_LIBS (no per-target probe).
    # Start generating CMakeLists.txt
    cat > "$target_dir/CMakeLists.txt" << 'CMAKEHEAD'
# =============================================================================
# Standalone CMakeLists.txt (auto-generated by extract_model_package.sh)
# =============================================================================
cmake_minimum_required(VERSION 3.14)
CMAKEHEAD

    cat >> "$target_dir/CMakeLists.txt" << EOF
project(${model_name}_standalone DESCRIPTION "${model_name} Standalone Example")
EOF

    cat >> "$target_dir/CMakeLists.txt" << 'CMAKEBODY'

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Find Required Packages
# =============================================================================
find_package(OpenCV REQUIRED)
# Async runners (common/runner/async_*.hpp) use std::thread, so pthread must be linked.
# The entry .cpp files do not mention threads themselves (the use is in the headers),
# so always link the standard Threads package.
find_package(Threads REQUIRED)

if(CROSS_COMPILE OR MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib/x64 REQUIRED)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib/x64)
else()
    find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
    set(DXRT_LIB dxrt)
endif()

# =============================================================================
# Include Directories
# =============================================================================
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/postprocess)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utility)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern)

# =============================================================================
# Common Libraries & Flags
# =============================================================================
set(COMMON_LIBS ${OpenCV_LIBS} ${DXRT_LIB} Threads::Threads)

# Filesystem library for older compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    list(APPEND COMMON_LIBS stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    list(APPEND COMMON_LIBS c++fs)
endif()

set(COMMON_FLAGS -Wall -Wextra -O3)

# PROJECT_ROOT_DIR: default to parent of this directory.
# Override with -DPROJECT_ROOT_DIR=/path/to/dx_app when building in a different location.
if(NOT DEFINED PROJECT_ROOT_OVERRIDE)
    get_filename_component(PROJECT_ROOT_OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()
add_compile_definitions(PROJECT_ROOT_DIR="${PROJECT_ROOT_OVERRIDE}")

# =============================================================================
# Source Files
# =============================================================================
set(UTILITY_SOURCES utility/common_util.cpp)
file(GLOB POSTPROCESS_SOURCES "postprocess/*.cpp")

CMAKEBODY

    # Generate target definitions
    for src in "${sync_sources[@]}"; do
        local target_name
        target_name="$(basename "${src%.cpp}")"
        cat >> "$target_dir/CMakeLists.txt" << EOF
# --- ${target_name} ---
add_executable(${target_name} ${src} \${UTILITY_SOURCES} \${POSTPROCESS_SOURCES})
target_compile_options(${target_name} PRIVATE \${COMMON_FLAGS})
target_link_libraries(${target_name} \${COMMON_LIBS})

EOF
    done

    for src in "${async_sources[@]}"; do
        local target_name
        target_name="$(basename "${src%.cpp}")"
        cat >> "$target_dir/CMakeLists.txt" << EOF
# --- ${target_name} ---
add_executable(${target_name} ${src} \${UTILITY_SOURCES} \${POSTPROCESS_SOURCES})
target_compile_options(${target_name} PRIVATE \${COMMON_FLAGS})
target_link_libraries(${target_name} \${COMMON_LIBS})

EOF
    done
}

# =========================================================================
# Resolve task/family or task/family/variant
# Prints "<family_rel>|<variant>" (variant empty for a whole family).
# =========================================================================
resolve_model_arg() {
    local lang="$1"
    local rel="${2%/}"
    local src_root
    case "$lang" in
        py)  src_root="$PY_SRC_DIR" ;;
        cpp) src_root="$CPP_SRC_DIR" ;;
        *)   return 1 ;;
    esac
    if [ -f "$src_root/$rel/config.json" ]; then
        echo "$(dirname "$rel")|$(basename "$rel")"
        return 0
    fi
    if [ -d "$src_root/$rel" ]; then
        echo "$rel|"
        return 0
    fi
    return 1
}

# Copy <variant>/config.json trees. only_variant empty means every model folder.
copy_model_configs() {
    local src_family="$1"
    local dst_family="$2"
    local only_variant="$3"
    local child name
    for child in "$src_family"/*/; do
        [ -d "$child" ] || continue
        [ -f "${child}config.json" ] || continue
        name="$(basename "$child")"
        if [ -n "$only_variant" ] && [ "$name" != "$only_variant" ]; then
            continue
        fi
        mkdir -p "$dst_family/$name"
        cp "${child}config.json" "$dst_family/$name/config.json"
    done
}

# =========================================================================
# C++ - Prepare a single model directory for standalone deployment
# =========================================================================
prepare_model_cpp() {
    local family_rel="$1"
    local only_variant="$2"
    local src_model_dir="$CPP_SRC_DIR/$family_rel"
    local target_dir
    local model_name
    model_name=$(basename "$family_rel")

    if [ ! -d "$src_model_dir" ]; then
        echo -e "${RED}[DXAPP] [ERROR]${NC} C++ directory not found: $family_rel"
        return 1
    fi

    # Each variant is its own example: factory + <variant>_sync.cpp + <variant>_async.cpp.
    if [ -n "$OUTPUT_DIR" ]; then
        copy_variant_example() {
            local src_variant="$1"
            local dst_variant="$2"
            local variant_name
            variant_name="$(basename "$src_variant")"
            mkdir -p "$dst_variant"
            cp "$src_variant/${variant_name}_sync.cpp" "$dst_variant/" 2>/dev/null || true
            cp "$src_variant/${variant_name}_async.cpp" "$dst_variant/" 2>/dev/null || true
            if [ -d "$src_variant/factory" ]; then
                cp -r "$src_variant/factory" "$dst_variant/factory"
            fi
            if [ -f "$src_variant/config.json" ]; then
                cp "$src_variant/config.json" "$dst_variant/config.json"
            fi
        }
        if [ -n "$only_variant" ]; then
            target_dir="$OUTPUT_DIR/cpp/$family_rel/$only_variant"
            copy_variant_example "$src_model_dir/$only_variant" "$target_dir"
        else
            target_dir="$OUTPUT_DIR/cpp/$family_rel"
            mkdir -p "$target_dir"
            for child in "$src_model_dir"/*/; do
                [ -d "$child" ] || continue
                [ -f "${child}config.json" ] || continue
                copy_variant_example "${child%/}" "$target_dir/$(basename "$child")"
            done
        fi
    else
        target_dir="$src_model_dir"
    fi

    local label="$family_rel"
    [ -n "$only_variant" ] && label="$family_rel/$only_variant"
    echo -e "${CYAN}[PREPARE]${NC} C++ $label${OUTPUT_DIR:+ → $target_dir}"

    # 1. Copy common/ headers
    rm -rf "$target_dir/common"
    cp -r "$CPP_COMMON_SRC" "$target_dir/common"
    echo "  → common/ copied"

    # 2. Postprocess headers included in common/
    echo "  → postprocess headers included in common/"

    # 3. Copy utility files
    rm -rf "$target_dir/utility"
    mkdir -p "$target_dir/utility"
    cp "$UTILITY_SRC"/common_util.cpp "$target_dir/utility/"
    cp "$UTILITY_SRC"/common_util.hpp "$target_dir/utility/"
    cp "$UTILITY_SRC"/common_util_inline.hpp "$target_dir/utility/" 2>/dev/null || true
    echo "  → utility/ copied"

    # 4. Copy extern files
    rm -rf "$target_dir/extern"
    mkdir -p "$target_dir/extern"
    cp "$EXTERN_SRC"/cxxopts.hpp "$target_dir/extern/"
    echo "  → extern/ copied"

    # 5. Copy external factory headers referenced by source files
    local ext_factory_count=0
    local scan_dir
    scan_dir="${OUTPUT_DIR:+$src_model_dir}"
    scan_dir="${scan_dir:-$target_dir}"
    while IFS= read -r inc_path; do
        [ -z "$inc_path" ] && continue
        local src_file="$CPP_SRC_DIR/$inc_path"
        if [ -f "$src_file" ]; then
            local inc_dir
            inc_dir=$(dirname "$inc_path")
            mkdir -p "$target_dir/$inc_dir"
            cp "$src_file" "$target_dir/$inc_path"
            ext_factory_count=$((ext_factory_count + 1))
        fi
    done < <(find "$scan_dir" -name '*.cpp' ! -path '*/common/*' -print0 2>/dev/null \
             | xargs -0 grep -h '#include "' 2>/dev/null \
             | sed -n 's/.*#include "\([^"]*\)".*/\1/p' \
             | grep -v '^common/' \
             | grep -v '^<' \
             | grep 'factory/' \
             | sort -u)
    if [ "$ext_factory_count" -gt 0 ]; then
        echo "  → external factories copied ($ext_factory_count files)"
    fi

    # 6. Prune unreachable common/ headers (opt-in)
    run_prune "$target_dir" cpp

    # 7. Generate standalone CMakeLists.txt
    generate_cmake "$target_dir" "$model_name"
    echo "  → CMakeLists.txt generated"

    echo -e "${GREEN}[SUCCESS]${NC} C++ standalone ready: $label"
    if [ -n "$OUTPUT_DIR" ]; then
        echo "  Build: cd $target_dir && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    else
        echo "  Build: cd $family_rel && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    fi
    echo ""
}

# =========================================================================
# Python - Remove standalone files
# =========================================================================
clean_model_py() {
    local model_dir="$1"
    local target_dir="$PY_SRC_DIR/$model_dir"

    if [ ! -d "$target_dir" ]; then
        echo -e "${RED}[DXAPP] [ERROR]${NC} Python directory not found: $model_dir"
        return 1
    fi

    rm -rf "$target_dir/common"
    echo -e "${GREEN}[CLEAN]${NC} Python $model_dir"
}

# =========================================================================
# Python - Prepare a single model directory for standalone deployment
# =========================================================================
prepare_model_py() {
    local family_rel="$1"
    local only_variant="$2"
    local src_model_dir="$PY_SRC_DIR/$family_rel"
    local target_dir

    if [ ! -d "$src_model_dir" ]; then
        echo -e "${RED}[DXAPP] [ERROR]${NC} Python directory not found: $family_rel"
        return 1
    fi

    # A single model is a self-contained folder: its scripts, config.json,
    # variant factory/, custom_ops.py and a pruned common/. A family export
    # keeps every model folder under <task>/<family>/.
    if [ -n "$OUTPUT_DIR" ]; then
        if [ -n "$only_variant" ]; then
            target_dir="$OUTPUT_DIR/py/$family_rel/$only_variant"
            mkdir -p "$target_dir"
            find "$src_model_dir/$only_variant" -maxdepth 1 -type f | while read -r f; do
                cp "$f" "$target_dir/"
            done
            if [ -d "$src_model_dir/$only_variant/factory" ]; then
                cp -r "$src_model_dir/$only_variant/factory" "$target_dir/factory"
            fi
        else
            target_dir="$OUTPUT_DIR/py/$family_rel"
            mkdir -p "$target_dir"
            find "$src_model_dir" -maxdepth 1 -type f | while read -r f; do
                cp "$f" "$target_dir/"
            done
            local child name
            for child in "$src_model_dir"/*/; do
                [ -f "${child}config.json" ] || continue
                name="$(basename "$child")"
                mkdir -p "$target_dir/$name"
                find "$child" -maxdepth 1 -type f | while read -r f; do
                    cp "$f" "$target_dir/$name/"
                done
                if [ -d "${child}factory" ]; then
                    rm -rf "$target_dir/$name/factory"
                    cp -r "${child}factory" "$target_dir/$name/factory"
                fi
            done
        fi
    elif [ -n "$only_variant" ]; then
        target_dir="$src_model_dir/$only_variant"
    else
        target_dir="$src_model_dir"
    fi

    local common_dst="$target_dir/common"

    # Remove existing common directory
    [ -d "$common_dst" ] && rm -rf "$common_dst"

    # Copy common directory
    cp -r "$PY_COMMON_SRC" "$common_dst"

    # Clean up __pycache__
    find "$common_dst" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    # Prune unreachable common/ modules and rewrite the __init__.py barrels (opt-in)
    run_prune "$target_dir" py

    local label="$family_rel"
    [ -n "$only_variant" ] && label="$family_rel/$only_variant"
    echo -e "${GREEN}[SUCCESS]${NC} Python standalone ready: $label${OUTPUT_DIR:+ → $target_dir}"
}

# =========================================================================
# Process a single model for the given language
# =========================================================================
process_model() {
    local model_dir="$1"
    local lang="$2"
    local resolved family_rel only_variant

    if ! resolved="$(resolve_model_arg "$lang" "$model_dir")"; then
        echo -e "${RED}[DXAPP] [ERROR]${NC} directory not found: $model_dir ($lang)"
        return 1
    fi
    family_rel="${resolved%%|*}"
    only_variant="${resolved#*|}"

    if [ "$CLEAN_MODE" = true ]; then
        case "$lang" in
            cpp) clean_model_cpp "$family_rel" ;;
            py)  clean_model_py "$family_rel" ;;
        esac
    else
        case "$lang" in
            cpp) prepare_model_cpp "$family_rel" "$only_variant" ;;
            py)  prepare_model_py "$family_rel" "$only_variant" ;;
        esac
    fi
}

# =========================================================================
# Discover all model directories for a language
# =========================================================================
discover_models() {
    local lang="$1"
    local src_dir

    case "$lang" in
        cpp) src_dir="$CPP_SRC_DIR" ;;
        py)  src_dir="$PY_SRC_DIR" ;;
    esac

    if [ "$lang" = "cpp" ]; then
        # C++: find directories containing *_sync.cpp
        find "$src_dir" -name "*_sync.cpp" -print0 2>/dev/null | while IFS= read -r -d '' src_file; do
            local model_dir="$(dirname "$src_file")"
            local rel_dir="${model_dir#$src_dir/}"
            case "$rel_dir" in
                common/*|build/*|.*) continue ;;
            esac
            echo "$rel_dir"
        done | sort -u
    else
        # Python: find directories containing factory/ subdirectory
        find "$src_dir" -type d -name "factory" ! -path "*/common/*" 2>/dev/null | sort | while read -r factory_dir; do
            local model_dir="$(dirname "$factory_dir")"
            local rel_dir="${model_dir#$src_dir/}"
            if [ "$rel_dir" != "common" ] && [ "$rel_dir" != "$src_dir" ]; then
                echo "$rel_dir"
            fi
        done
    fi
}

# =========================================================================
# Main
# =========================================================================
if [ "$MODEL_ARG" == "all" ]; then
    echo "=============================================="
    if [ "$CLEAN_MODE" = true ]; then
        echo " Cleaning all model directories (lang: $LANG_MODE)"
    else
        echo " Preparing all model directories for standalone deployment (lang: $LANG_MODE)"
        [ -n "$OUTPUT_DIR" ] && echo " Output: $OUTPUT_DIR"
    fi
    echo "=============================================="
    echo ""

    for lang in "${LANGS[@]}"; do
        LANG_LABEL=$(echo "$lang" | tr '[:lower:]' '[:upper:]')
        echo -e "${CYAN}━━━ $LANG_LABEL ━━━${NC}"

        discover_models "$lang" | while read -r rel_dir; do
            process_model "$rel_dir" "$lang"
        done
    done

    echo "=============================================="
    echo -e "${GREEN}[DONE]${NC} All model directories processed!"
    if [ -n "$OUTPUT_DIR" ]; then
        echo -e "${GREEN}[OUTPUT]${NC} $OUTPUT_DIR"
    fi
    echo "=============================================="
else
    # Strip trailing slash
    MODEL_ARG="${MODEL_ARG%/}"

    for lang in "${LANGS[@]}"; do
        process_model "$MODEL_ARG" "$lang"
    done

    if [ -n "$OUTPUT_DIR" ]; then
        echo -e "\n${GREEN}[OUTPUT]${NC} Exported to: $OUTPUT_DIR"
    fi
fi
