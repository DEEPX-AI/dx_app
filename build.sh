#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}")

# Calculate number of CPU cores to use (all but one to keep system responsive)
NUM_CORES=$(nproc)
BUILD_JOBS=$((NUM_CORES - 1))
# Ensure at least 1 job
if [ $BUILD_JOBS -lt 1 ]; then
    BUILD_JOBS=1
fi

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh
source ${SCRIPT_DIR}/scripts/build_target_resolver.sh

pushd ${DX_APP_PATH} >&2

help() {
    echo -e "Usage: ${COLOR_CYAN}$0 [OPTIONS]${COLOR_RESET}"
    echo -e "Build the project with various configuration options."
    echo -e ""
    echo -e "${COLOR_BOLD}Options:${COLOR_RESET}"
    echo -e "  ${COLOR_GREEN}--help${COLOR_RESET}       Display this help message and exit."
    echo -e "  ${COLOR_GREEN}--clean${COLOR_RESET}      Perform a clean build, removing previous build artifacts."
    echo -e "  ${COLOR_GREEN}--verbose${COLOR_RESET}    Show detailed build commands during the process."
    echo -e "  ${COLOR_GREEN}--type <TYPE>${COLOR_RESET}  Specify the CMake build type. Valid options: [Release, Debug, RelWithDebInfo]."
    echo -e "  ${COLOR_GREEN}--arch <ARCH>${COLOR_RESET}  Specify the target CPU architecture. Valid options: [x86_64, aarch64]."
    echo -e "  ${COLOR_GREEN}--target <NAME> [NAME2 ...]${COLOR_RESET} Build only the specified target(s) (e.g., yolov5_sync yolov5_async)."
    echo -e "                            Use '--target list' to show all available targets."
    echo -e "  ${COLOR_GREEN}--minimal${COLOR_RESET}     Build run_demo C++ sync/async targets only."
    echo -e "  ${COLOR_GREEN}--category <NAME|list>${COLOR_RESET} Build C++ sync/async targets under a task category."
    echo -e "  ${COLOR_GREEN}--make_so${COLOR_RESET}    Build postprocess shared library for dynamic linking (default: disabled)."
    echo -e "  ${COLOR_GREEN}--coverage${COLOR_RESET}   Enable code coverage reporting (adds --coverage flags)."
    echo -e ""
    echo -e "  ${COLOR_GREEN}--python_exec <PATH>${COLOR_RESET} Specify the Python executable to use for the build."
    echo -e "                            If omitted, the default system 'python3' will be used."
    echo -e "  ${COLOR_GREEN}--venv_path <PATH>${COLOR_RESET}  Specify the path to a virtual environment to activate for the build."
    echo -e "                            If omitted, no virtual environment will be activated."
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --type Release --arch x86_64${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --clean --verbose${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --target yolov5_sync --type debug${COLOR_RESET}  # Build only yolov5_sync"
    echo -e "  ${COLOR_YELLOW}$0 --target list${COLOR_RESET}                      # List available targets"
    echo -e "  ${COLOR_YELLOW}$0 --all --type Release${COLOR_RESET}                # Full build (all targets, installs to bin/)"
    echo -e "  ${COLOR_YELLOW}$0 --minimal --type Release${COLOR_RESET}            # Build run_demo C++ targets only"
    echo -e "  ${COLOR_YELLOW}$0 --category object_detection --type Release${COLOR_RESET}  # Build all C++ targets in a category"
    echo -e "  ${COLOR_YELLOW}$0 --category list${COLOR_RESET}                     # List available categories"
    echo -e ""
    echo -e "  ${COLOR_YELLOW}$0 --python_exec /usr/local/bin/python3.8${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path ./venv-dxnn${COLOR_RESET}"

    if [ "$1" == "error" ] && [[ ! -n "$2" ]]; then
        print_colored "Invalid or missing arguments." "ERROR"
        exit 1
    elif [ "$1" == "error" ] && [[ -n "$2" ]]; then
        print_colored "$2" "ERROR"
        exit 1
    elif [[ "$1" == "warn" ]] && [[ -n "$2" ]]; then
        print_colored "$2" "WARNING"
        return 0
    fi
    exit 0
}

# Helper: uninstall dx_postprocess and clean artifacts
uninstall_dx_postprocess() {
    echo -e "${TAG_INFO} Uninstalling dx_postprocess from the current Python environment if installed..."
    if ${python_exec} -m pip show dx_postprocess >/dev/null 2>&1; then
        ${python_exec} -m pip uninstall -y dx_postprocess || true
    else
        echo -e "${TAG_WARN} dx_postprocess is not installed (pip show returned non-zero)."
    fi
}

install_dx_postprocess_module() {
    if [ ! -d "src/bindings/python/dx_postprocess" ]; then
        return 0
    fi

    echo ""
    echo -e "${COLOR_CYAN}${COLOR_BOLD}Installing dx_postprocess Python module...${COLOR_RESET}"
    echo -e "${COLOR_CYAN}  → Python: ${python_exec}${COLOR_RESET}"
    echo -e "${COLOR_CYAN}  → Build type: ${build_type}${COLOR_RESET}"

    case "${build_type,,}" in
        "debug")
            cmake_build_type="Debug"
            strip_option="false"
            ;;
        "release")
            cmake_build_type="Release"
            strip_option="true"
            ;;
        "relwithdebinfo")
            cmake_build_type="RelWithDebInfo"
            strip_option="false"
            ;;
        *)
            cmake_build_type="Release"
            strip_option="true"
            ;;
    esac

    pushd "src/bindings/python/dx_postprocess" >/dev/null 2>&1

    if SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${cmake_build_type}" \
       SKBUILD_INSTALL_STRIP="${strip_option}" \
       PROJECT_ROOT="${DX_APP_PATH}" \
       ${python_exec} -m pip install . ; then
        echo -e "${COLOR_GREEN}${COLOR_BOLD}dx_postprocess installation completed successfully!${COLOR_RESET}"

        INSTALL_LOCATION=$(${python_exec} -c "import sys; print(sys.prefix)" 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo -e "${COLOR_GREEN}  ✓ Installed to: ${INSTALL_LOCATION}${COLOR_RESET}"
        fi
    else
        echo -e "${COLOR_RED}${COLOR_BOLD}dx_postprocess installation failed!${COLOR_RESET}"
        echo -e "${TAG_ERROR} Python module installation is required for complete build"
        popd >/dev/null 2>&1
        exit 1
    fi

    popd >/dev/null 2>&1
    echo ""
}

# cmake command
cmd=()
clean_build=false
verbose=false
target_arch=$(uname -m)
build_type=release  
build_gtest=false
build_with_codec=false
build_with_sharedlib=false
enable_coverage=false
build_target=""
build_targets=()
build_minimal=false
build_all=false
build_category=""
build_selection_count=0

# global variaibles
python_exec=""
python_exec_input=""
venv_path=""

[ $# -gt 0 ] && \
while (( $# )); do
    case "$1" in
        --help)  help; exit 0;;
        --clean) clean_build=true; shift;;
        --verbose) verbose=true; shift;;
        --type) 
            shift 
            build_type="${1,,}" 
            shift;;
        --arch)
            shift
            target_arch=$1
            shift;;
        --python_exec)
            shift
            python_exec_input=$1
            shift;;
        --venv_path)
            shift
            venv_path=$1
            shift;;
        --test)
            build_gtest=true;
            shift;;
        --make_so)
            build_with_sharedlib=true;
            shift;;
        --coverage)
            enable_coverage=true;
            shift;;
        --target)
            shift
            if [ -z "$1" ]; then
                echo "Error: No target specified. Use --target <target_name> [target_name2 ...]."
                exit 1
            fi
            # Consume all following non-option arguments as targets
            while [[ -n "$1" && "$1" != --* ]]; do
                build_targets+=("$1")
                shift
            done
            build_target="${build_targets[*]}"
            ;;
        --all)
            build_all=true
            shift;;
        --minimal)
            build_minimal=true
            shift;;
        --category)
            shift
            if [[ -z "$1" || "$1" == --* ]]; then
                echo -e "${TAG_ERROR} --category requires a category name or 'list'." >&2
                exit 1
            fi
            build_category="$1"
            shift;;
        --v3codec)
            build_with_codec=true;
            shift;;
        *)
            help "error" "Invalid argument : $1"
            exit 1;;
    esac
done

# Reject conflicting target selection modes
[ ${#build_targets[@]} -gt 0 ] && build_selection_count=$((build_selection_count + 1))
[ "${build_minimal}" = "true" ] && build_selection_count=$((build_selection_count + 1))
[ "${build_all}" = "true" ] && build_selection_count=$((build_selection_count + 1))
[ -n "${build_category}" ] && build_selection_count=$((build_selection_count + 1))
if [ "${build_selection_count}" -gt 1 ]; then
    echo -e "${TAG_ERROR} Use only one of --all, --minimal, --target, or --category." >&2
    exit 1
fi

# If no build mode was explicitly specified, default to minimal and notify the user
if [ "${build_selection_count}" -eq 0 ]; then
    echo -e "${TAG_INFO} No build mode specified."
    echo -e "${TAG_INFO} Available modes: ${COLOR_GREEN}--all${COLOR_RESET} | ${COLOR_GREEN}--minimal${COLOR_RESET} | ${COLOR_GREEN}--target <name>${COLOR_RESET} | ${COLOR_GREEN}--category <name>${COLOR_RESET}"
    echo -e "${TAG_INFO} Defaulting to ${COLOR_GREEN}--minimal${COLOR_RESET} build (run_demo C++ targets)."
    echo ""
    build_minimal=true
fi

# Handle --category list early (before Python/toolchain/CMake setup)
if [ "${build_category}" = "list" ]; then
    dxapp_list_categories
    exit 0
fi

# Check if venv_path
if [ -n "${venv_path}" ]; then
    if [ ! -f "${venv_path}/bin/python" ]; then
        echo -e "${TAG_ERROR} --venv_path is set to '${venv_path}'. but, Virtual environment path is invalid: ${venv_path}/bin/python. Please check the path." >&2
        exit 1
    else
        echo -e "${TAG_INFO} --venv_path is set to '${venv_path}'."
        . ${venv_path}/bin/activate;
    fi
fi

# Check if python_exec 
if [ -n "${python_exec_input}" ]; then
    if [ ! -f "${python_exec_input}" ]; then
        echo -e "${TAG_ERROR} --python_exec is set to '${python_exec_input}'. but, Python executable path does not exist: ${python_exec_input}. Please check the path." >&2
        exit 1
    else
        echo -e "${TAG_INFO} --python_exec is set to '${python_exec_input}'"
        ${python_exec_input} --version;
        python_exec=${python_exec_input}
    fi
else
    # use default python
    python_exec="python3"
fi

if [ $target_arch == "arm64" ]; then
    target_arch=aarch64
fi

cmd+=(-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.$target_arch.cmake)

dxrt_dir=$(grep -i ^set\(DXRT_INSTALLED_DIR cmake/toolchain.$target_arch.cmake | sed 's/set(DXRT_INSTALLED_DIR //' | sed 's/)//')
if [ -z "$dxrt_dir" ]; then
    dxrt_dir=/usr/local
fi
if [ ! -e "$dxrt_dir" ]; then
    echo -e "${TAG_ERROR} $dxrt_dir directory does not exist"
    exit -1
fi

if [ $build_gtest == "true" ]; then
    cmd+=(-DUSE_DXAPP_TEST=True);
fi

if [ $build_with_codec == "true" ]; then
    if [ ! -d "${DX_APP_PATH}/third_party/v3_codec" ]; then
        echo -e "${TAG_WARN} v3_codec directory not found at ${DX_APP_PATH}/third_party/v3_codec. Disabling codec build."
        build_with_codec=false
    else
        cmd+=(-DUSE_V3_CODEC=True);
        cmd+=(-DV3_CODEC_DIR=${DX_APP_PATH}/third_party/v3_codec);
    fi
fi

if [ $build_with_sharedlib == "true" ]; then
    cmd+=(-DDXAPP_WITH_SHAREDLIB=True);
fi

if [ $enable_coverage == "true" ]; then
    cmd+=(-DENABLE_COVERAGE=ON);
    echo -e "${TAG_INFO} Code coverage enabled"
fi

cmd+=(-DCMAKE_VERBOSE_MAKEFILE=$verbose)

if [ $build_type == "release" ] || [ $build_type == "debug" ] || [ $build_type == "relwithdebinfo" ]; then
    cmd+=(-DCMAKE_BUILD_TYPE=$build_type);
else
    cmd+=(-DCMAKE_BUILD_TYPE=release);
fi

cmd+=(-DCMAKE_GENERATOR=Ninja)

# Resolve target candidates if --minimal or --category is used
if [ "${build_minimal}" = "true" ]; then
    resolved_targets=$(dxapp_resolve_minimal_targets) || exit 1
    if [ -z "${resolved_targets}" ]; then
        echo -e "${TAG_ERROR} No build targets resolved." >&2
        exit 1
    fi
    mapfile -t build_targets <<< "${resolved_targets}"
elif [ -n "${build_category}" ]; then
    resolved_targets=$(dxapp_resolve_category_targets "${build_category}") || exit 1
    if [ -z "${resolved_targets}" ]; then
        echo -e "${TAG_ERROR} No build targets resolved." >&2
        exit 1
    fi
    mapfile -t build_targets <<< "${resolved_targets}"
fi

build_dir=build_"$target_arch"
out_dir=bin
echo cmake args : ${cmd[@]}

if [ $clean_build == "true" ]; then 
    # Uninstall python package and clean artifacts as part of clean build
    uninstall_dx_postprocess
    CLEAN_CMD="rm -rf $build_dir && [ $(uname -m) == \"$target_arch\" ] && rm -rf bin && rm -rf lib && rm -rf include"
    ${CLEAN_CMD}
    if [ $? -ne 0 ]; then
        echo -e "${TAG_WARN} Failed to clean build directory. try to clean again with 'sudo'."
        sudo ${CLEAN_CMD} 
        if [ $? -ne 0 ]; then
            echo -e "${TAG_ERROR} Failed to clean build directory"
            exit 1
        fi
    fi
fi

mkdir -p $build_dir 
rm -rf $build_dir/release 
pushd $build_dir >&2
cmake .. ${cmd[@]} || {
    echo -e "${TAG_ERROR} CMake configuration failed. Please check the output above."
    exit 1
}
echo -e "${TAG_INFO} Using $BUILD_JOBS parallel jobs (of $NUM_CORES available cores)"

# Filter resolved targets to only existing ones (for --minimal and --category)
filter_existing_targets() {
    local requested=("$@")
    local available
    local filtered=()
    mapfile -t available < <(ninja -t targets | sed 's/:.*//' | sort -u)
    for target in "${requested[@]}"; do
        if printf '%s\n' "${available[@]}" | grep -Fxq "${target}"; then
            filtered+=("${target}")
        else
            echo -e "${TAG_WARN} Build target not found, skipping: ${target}"
        fi
    done
    if [ ${#filtered[@]} -eq 0 ]; then
        echo -e "${TAG_ERROR} No build targets resolved." >&2
        exit 1
    fi
    build_targets=("${filtered[@]}")
}

# Handle --target list option
if [ "$build_target" == "list" ]; then
    echo -e "${COLOR_CYAN}${COLOR_BOLD}Available build targets:${COLOR_RESET}"
    ninja -t targets | grep -E "^[a-z].*: phony$" | grep -vE "(edit_cache|rebuild_cache|list_install_components|install)" | sed 's/: phony$//' | sort | column
    popd >/dev/null 2>&1
    exit 0
fi

# Filter targets if using --minimal or --category
if [ "${build_minimal}" = "true" ] || [ -n "${build_category}" ]; then
    filter_existing_targets "${build_targets[@]}"
fi

# Build specific target or all
if [ ${#build_targets[@]} -gt 0 ]; then
    echo -e "${TAG_INFO} Building targets: ${COLOR_GREEN}${build_targets[*]}${COLOR_RESET}"
    cmake --build . --target "${build_targets[@]}" --parallel $BUILD_JOBS || { echo -e "${TAG_ERROR} CMake build failed for target(s) '${build_targets[*]}'. Please check the output above."; exit 1; }

    # Copy the built binaries to bin directory if they exist
    if [ $(uname -m) == "$target_arch" ]; then
        mkdir -p ../bin
        for t in "${build_targets[@]}"; do
            local_bin=$(find . -name "${t}" -type f -executable 2>/dev/null | head -1)
            if [ -n "$local_bin" ]; then
                cp "$local_bin" ../bin/ 2>/dev/null || true
                echo -e "${TAG_INFO} Binary copied to bin/${t}"
            fi
        done
    fi
    popd >/dev/null 2>&1
elif [ $(uname -m) != "$target_arch" ]; then
    cmake --build . --target install --parallel $BUILD_JOBS || { echo -e "${TAG_ERROR} CMake build failed. Please check the output above."; exit 1; } && popd >/dev/null 2>&1
else
    cmake --build . --target install --parallel $BUILD_JOBS || { echo -e "${TAG_ERROR} CMake build failed. Please check the output above."; exit 1; } && popd >/dev/null 2>&1 && cp -r $build_dir/release/* ./
    if [ $? -ne 1 ]; then
        echo Build Completed and executable copied to $(pwd)
    fi
fi

# Install dx_postprocess (full pip build) for minimal builds; skip for explicit --target / --category builds
if [ ${#build_targets[@]} -gt 0 ]; then
    if [ "${build_minimal}" = "true" ]; then
        echo -e "${TAG_INFO} Building dx_postprocess Python bindings (full build)..."
        install_dx_postprocess_module
    fi
    echo -e "${TAG_INFO} Target build completed: ${COLOR_GREEN}${build_targets[*]}${COLOR_RESET}"
    echo ""
    exit 0
fi

if [ -e $build_dir/release/bin ]; then
    install_dx_postprocess_module

    echo Build Done. "($build_type)"
    echo =================================================
        echo clean_build : $clean_build
        echo verbose : $verbose
        echo build_type : $build_type
        echo target_arch : $target_arch
    echo =================================================    
    echo ""
    
    if [ "$build_with_sharedlib" == "true" ]; then
        # Interactive prompt for LD_LIBRARY_PATH setup (only when building shared lib)
        echo -e "${COLOR_CYAN}${COLOR_BOLD}Choose how to set up library path:${COLOR_RESET}"
        echo -e "  ${COLOR_GREEN}1)${COLOR_RESET} Export LD_LIBRARY_PATH (current session only, temporary)"
        echo -e "  ${COLOR_GREEN}2)${COLOR_RESET} Copy libs to /usr/local/lib and run ldconfig (system-wide, permanent)"
        echo ""
        echo -e "${COLOR_YELLOW}Press Enter for option 1 (default), or type 2 for option 2.${COLOR_RESET}"
        echo -e "${COLOR_YELLOW}No response within 10 seconds will skip both options.${COLOR_RESET}"
        echo ""
        
        # Read user input with 10 second timeout
        read -t 10 -p "Select option [1]: " user_choice
        read_status=$?
        
        if [ $read_status -gt 128 ]; then
            # Timeout occurred (no input within 10 seconds)
            echo ""
            echo -e "${COLOR_YELLOW}${COLOR_BOLD}Timeout: No option selected. Skipping library path setup.${COLOR_RESET}"
            echo -e "${COLOR_GREEN}${COLOR_BOLD}To manually set up later, use the provided scripts:${COLOR_RESET}"
            echo -e "${COLOR_GREEN}  - Temporary (session): ${COLOR_CYAN}source ./scripts/setup_postprocess_lib.sh --session${COLOR_RESET}"
            echo -e "${COLOR_GREEN}  - Permanent (system):  ${COLOR_CYAN}./scripts/setup_postprocess_lib.sh --system${COLOR_RESET}"
            echo ""
            echo -e "${COLOR_GREEN}${COLOR_BOLD}To remove the configuration:${COLOR_RESET}"
            echo -e "${COLOR_GREEN}  - Temporary (session): ${COLOR_CYAN}source ./scripts/unsetup_postprocess_lib.sh --session${COLOR_RESET}"
            echo -e "${COLOR_GREEN}  - Permanent (system):  ${COLOR_CYAN}./scripts/unsetup_postprocess_lib.sh --system${COLOR_RESET}"
        elif [ "$user_choice" == "2" ]; then
            # Option 2: Copy to /usr/local/lib and run ldconfig
            echo ""
            echo -e "${COLOR_CYAN}${COLOR_BOLD}Copying libraries to /usr/local/lib...${COLOR_RESET}"
            if sudo cp $(pwd)/lib/libdxapp_*_postprocess.so /usr/local/lib/ 2>/dev/null; then
                echo -e "${COLOR_GREEN}  ✓ Libraries copied to /usr/local/lib${COLOR_RESET}"
                echo -e "${COLOR_CYAN}${COLOR_BOLD}Running ldconfig...${COLOR_RESET}"
                if sudo ldconfig; then
                    echo -e "${COLOR_GREEN}  ✓ ldconfig completed successfully${COLOR_RESET}"
                    echo -e "${COLOR_GREEN}${COLOR_BOLD}Library path is now permanently configured (system-wide).${COLOR_RESET}"
                else
                    echo -e "${COLOR_RED}${COLOR_BOLD}Failed to run ldconfig.${COLOR_RESET}"
                fi
            else
                echo -e "${COLOR_RED}${COLOR_BOLD}Failed to copy libraries to /usr/local/lib.${COLOR_RESET}"
                echo -e "${TAG_WARN} You may need to manually run: sudo cp $(pwd)/lib/libdxapp_*_postprocess.so /usr/local/lib && sudo ldconfig"
            fi
        else
            # Option 1 (default): Export LD_LIBRARY_PATH
            echo ""
            echo -e "${COLOR_CYAN}${COLOR_BOLD}Setting LD_LIBRARY_PATH for current session...${COLOR_RESET}"
            export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
            echo -e "${COLOR_GREEN}  ✓ LD_LIBRARY_PATH exported: $(pwd)/lib${COLOR_RESET}"
            echo -e "${COLOR_YELLOW}${COLOR_BOLD}Note: This setting is temporary and will be lost when the terminal session ends.${COLOR_RESET}"
            echo -e "${COLOR_YELLOW}To make it permanent, add the following line to your ~/.bashrc or ~/.profile:${COLOR_RESET}"
            echo -e "${COLOR_GREEN}  export LD_LIBRARY_PATH=$(pwd)/lib:\$LD_LIBRARY_PATH${COLOR_RESET}"
        fi
        echo ""
    fi
else
    echo Build Failed.
    exit -1
fi

popd >/dev/null 2>&1
