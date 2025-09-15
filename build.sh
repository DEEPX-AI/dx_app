#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}")

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

    echo -e "${TAG_INFO} Removing dx_postprocess artifacts from current interpreter's site/dist-packages..."
    SITE_PKGS=$(${python_exec} - <<'PY'
import site, sys, sysconfig, os
roots=set()
paths = sysconfig.get_paths() or {}
for k in ('purelib','platlib'):
    p = paths.get(k)
    if p:
        roots.add(p)
try:
    up = site.getusersitepackages()
    if up:
        roots.add(up)
except Exception:
    pass
prefix = os.path.realpath(sys.prefix)
filtered = [r for r in roots if os.path.realpath(r).startswith(prefix)]
print("\n".join(sorted(filtered)))
PY
)
    for d in ${SITE_PKGS}; do
        [ -d "$d" ] || continue
        rm -f "$d"/dx_postprocess*.so "$d"/dx_postprocess*.pyd 2>/dev/null || true
        rm -rf "$d"/dx_postprocess-*.dist-info "$d"/dx_postprocess*.egg-info "$d"/dx_postprocess 2>/dev/null || true
    done

    echo -e "${TAG_INFO} Removing local build artifacts for dx_postprocess..."
    rm -rf lib/pybind/build lib/pybind/dist lib/pybind/dx_postprocess.egg-info 2>/dev/null || true
}

# cmake command
cmd=()
clean_build=false
verbose=false
target_arch=$(uname -m)
build_type=release  
build_gtest=false

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
            python_exec_input=$(realpath "$1")
            shift;;
        --venv_path)
            shift
            venv_path=$1
            shift;;
        --test)
            build_gtest=true;
            shift;;
        *)
            help "error" "Invalid argument : $1"
            exit 1;;
    esac
done

# this function is defined in scripts/common_util.sh
# Usage: os_check "supported_os_names" "ubuntu_versions" "debian_versions"
os_check "ubuntu debian" "18.04 20.04 22.04 24.04" "12"

# this function is defined in scripts/common_util.sh
# Usage: arch_check "supported_arch_names"
arch_check "amd64 x86_64 arm64 aarch64 armv7l"

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
        echo -e "${TAG_ERROR} --python_exec is set to '${python_exec_input}'. but, Python executable path does not exist: ${python_exec}. Please check the path." >&2
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
    CLEAN_CMD="rm -rf $build_dir && [ $(uname -m) == \"$target_arch\" ] && rm -rf $out_dir"
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
mkdir -p $out_dir
rm -rf $build_dir/release/bin
pushd $build_dir >&2
cmake .. ${cmd[@]} || {
    echo -e "${TAG_ERROR} CMake configuration failed. Please check the output above."
    exit 1
}
if [ $(uname -m) != "$target_arch" ]; then
    cmake --build . --target install || { echo -e "${TAG_ERROR} CMake build failed. Please check the output above."; exit 1; } && popd >&2
else
    cmake --build . --target install || { echo -e "${TAG_ERROR} CMake build failed. Please check the output above."; exit 1; } && popd >&2 && cp $build_dir/release/bin/* $out_dir/
    if [ $? -ne 1 ]; then
        echo Build Completed and executable copied to $out_dir/
    fi
fi

if [ -f ${DX_APP_PATH}/templates/python/requirements.txt ]; then
    echo Installing Python dependencies from ${DX_APP_PATH}/templates/python/requirements.txt
    pip install -r ${DX_APP_PATH}/templates/python/requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${TAG_ERROR} Failed to install Python dependencies"
        exit 1
    fi
else
    echo -e "${TAG_ERROR} ${DX_APP_PATH}/templates/python/requirements.txt not found"
    exit 1
fi

# Install dx_postprocess Python module
if [ -f ${DX_APP_PATH}/lib/pybind/setup.py ]; then
    echo "Installing dx_postprocess Python module..."
    pushd ${DX_APP_PATH}/lib/pybind >&2
    
    if [ $build_type == "debug" ]; then
        export CMAKE_BUILD_TYPE=Debug
        export DEBUG=1
        echo "Setting DEBUG mode for dx_postprocess build"
    elif [ $build_type == "relwithdebinfo" ]; then
        export CMAKE_BUILD_TYPE=RelWithDebInfo
        export DEBUG=1
        echo "Setting RelWithDebInfo mode for dx_postprocess build"
    else
        export CMAKE_BUILD_TYPE=Release
        export DEBUG=0
        echo "Setting RELEASE mode for dx_postprocess build"
    fi
    
    ${python_exec} -m pip install .
    install_result=$?
    popd >&2
    if [ $install_result -eq 0 ]; then
        echo "dx_postprocess module installed successfully"
        # Verify installation
        ${python_exec} -c "import dx_postprocess; print('dx_postprocess module verification: OK')" 2>/dev/null;
        if [ $? -ne 0 ]; then
            echo -e "${TAG_ERROR} dx_postprocess module import failed"
            exit 1
        fi 
    else
        echo -e "${TAG_ERROR} Failed to install dx_postprocess module"
        exit 1
    fi

    if [ -n "$venv_path" ]; then
        echo -e "${TAG_INFO} To activate the virtual environment, run:"
        echo -e "${COLOR_BRIGHT_YELLOW_ON_BLACK}  source ${venv_path}/bin/activate ${COLOR_RESET}"
    fi
else
    echo -e "${TAG_ERROR} ./lib/pybind/setup.py not found"
    exit 1
fi

if [ -e $build_dir/release/bin ]; then
    echo Build Done. "($build_type)"
    echo =================================================
        echo clean_build : $clean_build
        echo verbose : $verbose
        echo build_type : $build_type
        echo target_arch : $target_arch
    echo =================================================
else
    echo Build Failed.
    exit -1
fi

popd >&2
