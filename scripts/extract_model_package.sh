#!/bin/bash
# =============================================================================
# extract_model_package.sh - Prepare model directories for standalone deployment (C++/Python)
# =============================================================================
# Usage:
#   ./extract_model_package.sh <model_dir|all> [--lang cpp|py|both] [--output-dir <path>] [--clean]
#
# C++ mode:  Copy common/, utility/, extern/, generate CMakeLists.txt
# Python mode: Copy common/
#
# Options:
#   --output-dir <path>  Export standalone package to a separate directory
#                        instead of modifying the source tree in-place.
#                        Creates <path>/<lang>/<category>/<model>/
#
# Examples:
#   ./extract_model_package.sh object_detection/yolov7
#   ./extract_model_package.sh all
#   ./extract_model_package.sh all --lang cpp
#   ./extract_model_package.sh all --lang py
#   ./extract_model_package.sh object_detection/yolov7 --clean
#   ./extract_model_package.sh all --clean --lang both
#   ./extract_model_package.sh object_detection/yolov7 --output-dir outputs/
#   ./extract_model_package.sh all --output-dir /tmp/standalone --lang cpp

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
    echo "Usage: $0 <model_dir|all> [--lang cpp|py|both] [--output-dir <path>] [--clean]"
    echo ""
    echo "Arguments:"
    echo "  model_dir           Relative path to model directory (e.g., object_detection/yolov7)"
    echo "  all                 Prepare all model directories"
    echo "  --lang <mode>       Language: cpp, py, or both (default: both)"
    echo "  --output-dir <path> Export to a separate directory (default: in-place)"
    echo "  --clean             Remove standalone files instead of creating them"
    echo ""
    echo "Examples:"
    echo "  $0 object_detection/yolov7"
    echo "  $0 all --lang cpp"
    echo "  $0 object_detection/yolov8 --lang py"
    echo "  $0 all --clean"
    echo "  $0 all --output-dir outputs/"
    echo "  $0 object_detection/yolov7 --output-dir /tmp/deploy --lang cpp"
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

while [ $# -gt 0 ]; do
    case "$1" in
        --clean)      CLEAN_MODE=true;  shift ;;
        --lang)       LANG_MODE="$2";   shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";  shift 2 ;;
        -h|--help)    usage ;;
        *)            MODEL_ARG="$1";   shift ;;
    esac
done

if [ -z "$MODEL_ARG" ]; then usage; fi

# Resolve output-dir to absolute path
if [ -n "$OUTPUT_DIR" ]; then
    if [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="$(cd "$DX_APP_ROOT" && pwd)/$OUTPUT_DIR"
    fi
    if [ "$CLEAN_MODE" = true ]; then
        echo -e "${YELLOW}[WARN]${NC} --clean is ignored with --output-dir"
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
        echo -e "${RED}[ERROR]${NC} C++ directory not found: $model_dir"
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
        sync_sources+=("$(basename "$f")")
    done < <(find "$target_dir" -maxdepth 1 -name "*_sync.cpp" -print0 2>/dev/null)
    while IFS= read -r -d '' f; do
        async_sources+=("$(basename "$f")")
    done < <(find "$target_dir" -maxdepth 1 -name "*_async.cpp" -print0 2>/dev/null)

    # Check if any async target needs pthread
    local needs_pthread=false
    for src in "${async_sources[@]}"; do
        if grep -q 'std::thread\|pthread' "$target_dir/$src" 2>/dev/null; then
            needs_pthread=true
            break
        fi
    done

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

if(CROSS_COMPILE OR MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib)
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
set(COMMON_LIBS ${OpenCV_LIBS} ${DXRT_LIB})

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
        local target_name="${src%.cpp}"
        cat >> "$target_dir/CMakeLists.txt" << EOF
# --- ${target_name} ---
add_executable(${target_name} ${src} \${UTILITY_SOURCES} \${POSTPROCESS_SOURCES})
target_compile_options(${target_name} PRIVATE \${COMMON_FLAGS})
target_link_libraries(${target_name} \${COMMON_LIBS})

EOF
    done

    for src in "${async_sources[@]}"; do
        local target_name="${src%.cpp}"
        local extra_libs=""
        if [ "$needs_pthread" = true ]; then
            extra_libs=" pthread"
        fi
        cat >> "$target_dir/CMakeLists.txt" << EOF
# --- ${target_name} ---
add_executable(${target_name} ${src} \${UTILITY_SOURCES} \${POSTPROCESS_SOURCES})
target_compile_options(${target_name} PRIVATE \${COMMON_FLAGS})
target_link_libraries(${target_name} \${COMMON_LIBS}${extra_libs})

EOF
    done
}

# =========================================================================
# C++ - Prepare a single model directory for standalone deployment
# =========================================================================
prepare_model_cpp() {
    local model_dir="$1"
    local src_model_dir="$CPP_SRC_DIR/$model_dir"
    local target_dir
    local model_name
    model_name=$(basename "$model_dir")

    if [ ! -d "$src_model_dir" ]; then
        echo -e "${RED}[ERROR]${NC} C++ directory not found: $model_dir"
        return 1
    fi

    # Determine target: output-dir or in-place
    if [ -n "$OUTPUT_DIR" ]; then
        target_dir="$OUTPUT_DIR/cpp/$model_dir"
        mkdir -p "$target_dir"
        # Copy model source files
        find "$src_model_dir" -maxdepth 1 -type f | while read -r f; do
            cp "$f" "$target_dir/"
        done
        # Copy factory/ subdirectory
        if [ -d "$src_model_dir/factory" ]; then
            cp -r "$src_model_dir/factory" "$target_dir/factory"
        fi
    else
        target_dir="$src_model_dir"
    fi

    echo -e "${CYAN}[PREPARE]${NC} C++ $model_dir${OUTPUT_DIR:+ → $target_dir}"

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
    done < <(grep -rh '#include "' "$scan_dir"/*.cpp 2>/dev/null \
             | sed -n 's/.*#include "\([^"]*\)".*/\1/p' \
             | grep -v '^common/' \
             | grep -v '^<' \
             | grep 'factory/' \
             | sort -u)
    if [ "$ext_factory_count" -gt 0 ]; then
        echo "  → external factories copied ($ext_factory_count files)"
    fi

    # 6. Generate standalone CMakeLists.txt
    generate_cmake "$target_dir" "$model_name"
    echo "  → CMakeLists.txt generated"

    echo -e "${GREEN}[SUCCESS]${NC} C++ standalone ready: $model_dir"
    if [ -n "$OUTPUT_DIR" ]; then
        echo "  Build: cd $target_dir && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    else
        echo "  Build: cd $model_dir && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
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
        echo -e "${RED}[ERROR]${NC} Python directory not found: $model_dir"
        return 1
    fi

    rm -rf "$target_dir/common"
    echo -e "${GREEN}[CLEAN]${NC} Python $model_dir"
}

# =========================================================================
# Python - Prepare a single model directory for standalone deployment
# =========================================================================
prepare_model_py() {
    local model_dir="$1"
    local src_model_dir="$PY_SRC_DIR/$model_dir"
    local target_dir

    if [ ! -d "$src_model_dir" ]; then
        echo -e "${RED}[ERROR]${NC} Python directory not found: $model_dir"
        return 1
    fi

    # Determine target: output-dir or in-place
    if [ -n "$OUTPUT_DIR" ]; then
        target_dir="$OUTPUT_DIR/py/$model_dir"
        mkdir -p "$target_dir"
        # Copy model source files
        find "$src_model_dir" -maxdepth 1 -type f | while read -r f; do
            cp "$f" "$target_dir/"
        done
        # Copy factory/ subdirectory
        if [ -d "$src_model_dir/factory" ]; then
            cp -r "$src_model_dir/factory" "$target_dir/factory"
        fi
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

    echo -e "${GREEN}[SUCCESS]${NC} Python standalone ready: $model_dir${OUTPUT_DIR:+ → $target_dir}"
}

# =========================================================================
# Process a single model for the given language
# =========================================================================
process_model() {
    local model_dir="$1"
    local lang="$2"

    if [ "$CLEAN_MODE" = true ]; then
        case "$lang" in
            cpp) clean_model_cpp "$model_dir" ;;
            py)  clean_model_py "$model_dir" ;;
        esac
    else
        case "$lang" in
            cpp) prepare_model_cpp "$model_dir" ;;
            py)  prepare_model_py "$model_dir" ;;
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
    [ -n "$OUTPUT_DIR" ] && echo -e "${GREEN}[OUTPUT]${NC} $OUTPUT_DIR"
    echo "=============================================="
else
    # Strip trailing slash
    MODEL_ARG="${MODEL_ARG%/}"

    for lang in "${LANGS[@]}"; do
        process_model "$MODEL_ARG" "$lang"
    done

    [ -n "$OUTPUT_DIR" ] && echo -e "\n${GREEN}[OUTPUT]${NC} Exported to: $OUTPUT_DIR"
fi
