#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

# color env settings
source "${SCRIPT_DIR}/scripts/color_env.sh"

help() {
    echo -e "Usage: ${COLOR_CYAN}$0 [OPTIONS]${COLOR_RESET}"
    echo -e "Build the project with various configuration options."
    echo -e ""
    echo -e "${COLOR_BOLD}Options:${COLOR_RESET}"
    echo -e "  ${COLOR_GREEN}--help${COLOR_RESET}       Display this help message and exit."
    echo -e "  ${COLOR_GREEN}--clean${COLOR_RESET}      Perform a clean build, removing previous build artifacts."
    echo -e "  ${COLOR_GREEN}--verbose${COLOR_RESET}    Show detailed build commands during the process."
    echo -e "  ${COLOR_GREEN}--type <TYPE>${COLOR_RESET}  Specify the CMake build type. Valid options: [Release, Debug, RelWithDebInfo]."
    echo -e "  ${COLOR_GREEN}--arch <ARCH>${COLOR_RESET}  Specify the target CPU architecture. Valid options: [x86_64, aarch64, riscv64]."
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
    exit 0
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
            python_exec=$(realpath "$1")
            shift;;
        --venv_path)
            shift
            venv_path=$1
            shift;;
        --test) build_gtest=true; shift;;
        *)       echo "Invalid argument : " $1 ; help; exit -1;;
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

# Check if venv_path
if [ -n "${python_exec}" ]; then
    if [ ! -f "${python_exec}" ]; then
        echo -e "${TAG_ERROR} --python_exec is set to '${python_exec}'. but, Python executable path does not exist: ${python_exec}. Please check the path." >&2
        exit 1
    else
        echo -e "${TAG_INFO} --python_exec is set to '${python_exec}'"
        ${python_exec} --version;
    fi
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
    echo "-- Error : $dxrt_dir directory does not exist"
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
[ $clean_build == "true" ] && rm -rf $build_dir && [ $(uname -m) == "$target_arch" ] && rm -rf $out_dir/

mkdir -p $build_dir
mkdir -p $out_dir
rm -rf $build_dir/release/bin
cd $build_dir
cmake .. ${cmd[@]}
if [ $(uname -m) != "$target_arch" ]; then
    cmake --build . --target install && cd ..
else
    cmake --build . --target install && cd .. && cp $build_dir/release/bin/* $out_dir/
    if [ $? -ne 1 ]; then
        echo Build Completed and executable copied to $out_dir/
    fi
fi

if [ -f ./templates/python/requirements.txt ]; then
    echo Installing Python dependencies from ./templates/python/requirements.txt
    pip install -r ./templates/python/requirements.txt
    if [ $? -ne 0 ]; then
        echo "-- Warning: Failed to install Python dependencies"
    fi
else
    echo "-- Warning: ./templates/python/requirements.txt not found"
fi

# Install dx_postprocess Python module
if [ -f ./lib/pybind/setup.py ]; then
    echo "Installing dx_postprocess Python module..."
    cd ./lib/pybind
    ${python_exec} -m pip install .
    install_result=$?
    cd ../..
    if [ $install_result -eq 0 ]; then
        echo "dx_postprocess module installed successfully"
        # Verify installation
        python -c "import dx_postprocess; print('dx_postprocess module verification: OK')" 2>/dev/null || echo "-- Warning: dx_postprocess module import failed"
    else
        echo "-- Warning: Failed to install dx_postprocess module"
    fi

    if [ -n "$venv_path" ]; then
        echo -e "${TAG_INFO} To activate the virtual environment, run:"
        echo -e "${COLOR_BRIGHT_YELLOW_ON_BLACK}  source ${venv_path}/bin/activate ${COLOR_RESET}"
    fi
else
    echo "-- Warning: ./lib/pybind/setup.py not found"
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