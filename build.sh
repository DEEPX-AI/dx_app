#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}")

# Calculate number of CPU cores to use (half of available cores)
NUM_CORES=$(nproc)
BUILD_JOBS=$((NUM_CORES / 2))
# Ensure at least 1 job
if [ $BUILD_JOBS -lt 1 ]; then
    BUILD_JOBS=1
fi

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

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
    echo -e ""
    echo -e "  ${COLOR_GREEN}--python_exec <PATH>${COLOR_RESET} Specify the Python executable to use for the build."
    echo -e "                            If omitted, the default system 'python3' will be used."
    echo -e "  ${COLOR_GREEN}--venv_path <PATH>${COLOR_RESET}  Specify the path to a virtual environment to activate for the build."
    echo -e "                            If omitted, no virtual environment will be activated."
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --type Release --arch x86_64${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --clean --verbose${COLOR_RESET}"
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

# cmake command
cmd=()
clean_build=false
verbose=false
target_arch=$(uname -m)
build_type=release  
build_gtest=false
build_with_codec=false

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
        --v3codec)
            build_with_codec=true;
            shift;;
        *)
            help "error" "Invalid argument : $1"
            exit 1;;
    esac
done

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
if [ $dxrt_dir == "" ]; then
    dxrt_dir=/usr/local
fi
if [ ! -e $dxrt_dir ]; then
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

cmd+=(-DCMAKE_VERBOSE_MAKEFILE=$verbose)

if [ $build_type == "release" ] || [ $build_type == "debug" ] || [ $build_type == "relwithdebinfo" ]; then
    cmd+=(-DCMAKE_BUILD_TYPE=$build_type);
else
    cmd+=(-DCMAKE_BUILD_TYPE=release);
fi

cmd+=(-DCMAKE_GENERATOR=Ninja)

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
echo -e "${TAG_INFO} Using $BUILD_JOBS parallel jobs (half of $NUM_CORES available cores)"

if [ $(uname -m) != "$target_arch" ]; then
    cmake --build . --target install --parallel $BUILD_JOBS || { echo -e "${TAG_ERROR} CMake build failed. Please check the output above."; exit 1; } && popd >/dev/null 2>&1
else
    cmake --build . --target install --parallel $BUILD_JOBS || { echo -e "${TAG_ERROR} CMake build failed. Please check the output above."; exit 1; } && popd >/dev/null 2>&1 && cp -r $build_dir/release/* ./
    if [ $? -ne 1 ]; then
        echo Build Completed and executable copied to $(pwd)
    fi
fi

if [ -e $build_dir/release/bin ]; then
    # Install dx_postprocess Python module if available
    if [ -d "src/bindings/python/dx_postprocess" ]; then
        echo ""
        echo -e "${COLOR_CYAN}${COLOR_BOLD}Installing dx_postprocess Python module...${COLOR_RESET}"
        echo -e "${COLOR_CYAN}  → Python: ${python_exec}${COLOR_RESET}"
        echo -e "${COLOR_CYAN}  → Build type: ${build_type}${COLOR_RESET}"
        
        # Set build type for CMake (normalize to CMake format)
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
    fi

    echo Build Done. "($build_type)"
    echo =================================================
        echo clean_build : $clean_build
        echo verbose : $verbose
        echo build_type : $build_type
        echo target_arch : $target_arch
    echo =================================================    
    echo ""
    echo -e "${COLOR_GREEN}${COLOR_BOLD}please export LD_LIBRARY_PATH to $(pwd)/lib${COLOR_RESET}"
    echo -e "${COLOR_GREEN}${COLOR_BOLD}insert command line : \"export LD_LIBRARY_PATH=\$(pwd)/lib:\$LD_LIBRARY_PATH\"${COLOR_RESET}"
    echo ""
else
    echo Build Failed.
    exit -1
fi

popd >/dev/null 2>&1