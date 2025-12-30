#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}")

# variables for venv options
PYTHON_VERSION=""
VENV_PATH=""
VENV_FORCE_REMOVE="n"
VENV_REUSE="n"

ENABLE_DEBUG_LOGS=0

# Global variables for script configuration
MIN_PY_VERSION="3.11.0"

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

# dependencies install script in host
pushd ${DX_APP_PATH} >&2
cmd=()
DX_SRC_DIR=$PWD
echo "DX_SRC_DIR the default one $DX_SRC_DIR"
target_arch=$(uname -m)  
install_opencv=false  
install_dep=false  
opencv_source_build=false  
build_type='Release'

export DEBIAN_FRONTEND=noninteractive

function help() {
    echo -e "Usage: ${COLOR_CYAN}$0 [OPTIONS]${COLOR_RESET}"
    echo -e "Install necessary components and libraries for the project."
    echo -e ""
    echo -e "${COLOR_BOLD}Options:${COLOR_RESET}"
    echo -e "  ${COLOR_GREEN}--arch <ARCH>${COLOR_RESET}            Specify the target CPU architecture. Valid options: [x86_64, aarch64]."
    echo -e "  ${COLOR_GREEN}--dep${COLOR_RESET}                    Install core dependencies such as CMake, GCC, Ninja, etc."
    echo -e "  ${COLOR_GREEN}--opencv${COLOR_RESET}                 (Optional) Install the OpenCV package using system packages."
    echo -e "  ${COLOR_GREEN}--opencv-source-build${COLOR_RESET}    (Optional) Install the OpenCV package by compiling from source."
    echo -e "  ${COLOR_GREEN}--all${COLOR_RESET}                    Install all dependencies and the OpenCV package (via system packages)."
    echo -e ""
    echo -e "  ${COLOR_GREEN}--python_version <VERSION>${COLOR_RESET}   Specify the Python version to install (e.g., 3.10.4)."
    echo -e "                                 * Minimum supported version: ${MIN_PY_VERSION}."
    echo -e "                                 * If not specified:"
    echo -e "                                     - For Ubuntu 20.04+, the OS default Python 3 will be used."
    echo -e "                                     - For Ubuntu 18.04, Python ${MIN_PY_VERSION} will be source-built."
    echo -e "  ${COLOR_GREEN}--venv_path <PATH>${COLOR_RESET}          Specify the path for the virtual environment."
    echo -e "                                 * If this option is omitted, no virtual environment will be created."
    echo -e "  ${COLOR_GREEN}--help${COLOR_RESET}                      Display this help message and exit."
    echo -e ""
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --arch x86_64 --dep${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --all${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --opencv-source-build${COLOR_RESET}"
    echo -e ""
    echo -e "  ${COLOR_YELLOW}$0 --python_version 3.10.4 --venv_path /opt/my_venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --python_version 3.9.18  # Installs Python, but no venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path ./venv-dxnn # Installs default Python, creates venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 # Installs default Python, but no venv${COLOR_RESET}"
    echo -e ""

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

function compare_version() {
    awk -v n1="$1" -v n2="$2" 'BEGIN { if (n1 >= n2) exit 0; else exit 1; }'
}

function install_dep() {
    cmake_version_required=3.14
    install_cmake=false
    if [ "$install_dep" == true ]; then
        echo " Install dependence package tools "
        sudo apt-get update 
        if [ $? -ne 0 ]; then
            echo "Failed to apt update."
            exit 1
        fi
        sudo apt-get update && sudo apt-get -y install build-essential make zlib1g-dev libcurl4-openssl-dev wget tar zip cmake
        echo ""
        echo " Install python libraries" 
        sudo apt-get -y install python3-dev python3-setuptools python3-pip python3-tk python3-lxml python3-six python3-venv lsb-release

        echo " pybind11 will be downloaded via git clone during build process"

        cmake_version=$(cmake --version |grep -oP "\d+\.\d+\.\d+")
        if compare_version "$cmake_version" "$cmake_version_required"; then
            install_cmake=false
        else
            install_cmake=true
        fi
        if [ "$install_cmake" == true ]; then
            if ! test -e $DX_SRC_DIR/util; then 
                mkdir $DX_SRC_DIR/util
            fi
            cd $DX_SRC_DIR/util
            if ! test -e $DX_SRC_DIR/util/cmake-$cmake_version_required.0; then
                echo " Install CMake v$$cmake_version_required.0 "
                wget https://cmake.org/files/v$cmake_version_required/cmake-$cmake_version_required.0.tar.gz --no-check-certificate    
                tar xvf cmake-$cmake_version_required.0.tar.gz
            else
                echo " Already Exist CMake "
            fi
            cd cmake-$cmake_version_required.0
            ./bootstrap --system-curl
            make -j $(($(nproc) / 2))
            sudo make install 
        fi
        sudo apt install ninja-build
        sudo apt-get -y install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    fi

    install_python
}

function install_opencv() {
    cross_compile_opencv_mode=false
    if [ "$install_opencv" == true ] || [ "$opencv_source_build" == true ]; then
        toolchain_define="-D CMAKE_INSTALL_PREFIX=/usr/local "
        if [ $(uname -m) != "$target_arch" ]; then
            case "$target_arch" in
              arm64) toolchain_define="-D CMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake -D CMAKE_INSTALL_PREFIX=$DX_SRC_DIR/third_party/$target_arch "
              ;;
              aarch64) toolchain_define="-D CMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake -D CMAKE_INSTALL_PREFIX=$DX_SRC_DIR/third_party/$target_arch "
              ;;
            esac  
            if [ $(uname -m) == "arm64" ] && [ $target_arch == "aarch64" ]; then  
                toolchain_define="-D CMAKE_INSTALL_PREFIX=/usr/local "
            else
                echo " OpenCV Cross Compilation "
                manually_opencv_install=true
                cross_compile_opencv_mode=true
            fi
        fi
        if [ "$opencv_source_build" == true ]; then
            manually_opencv_install=true
        fi
        echo " Install opencv dependent library "
        sudo apt -y install libjpeg-dev libtiff5-dev ffmpeg \
             libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libavutil-dev \
             libtbb-dev libeigen3-dev libx264-dev libv4l-dev v4l-utils
        
        if apt-cache show libfreetype-dev > /dev/null 2>&1; then
            sudo apt-get install -y libfreetype-dev
        fi

        sudo apt-get clean && sudo apt update && sudo apt-get -y upgrade
        sudo apt -y install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev 

        sudo apt-get clean && sudo apt update && sudo apt-get -y upgrade 
        sudo apt-get -y install libgtk2.0-dev

        if [ -z $manually_opencv_install ]; then 
            sudo apt -y --reinstall install libopencv-dev python3-opencv
        fi

        if [ $? -ne 0 ]; then
            echo "Failed to install OpenCV dependent libraries."
            exit 1
        fi

        # Get the installed OpenCV version
        installed_version=$(opencv_version)

        # Define the minimum required version
        minimum_version="4.2.0"
        suggestion_version="4.5.5"

        # Compare versions using sort -V
        if [ "$(printf '%s\n' "$minimum_version" "$installed_version" | sort -V | head -n 1)" == "$minimum_version" ] && [ -z $manually_opencv_install ]; then
            if [ "$installed_version" == "$minimum_version" ]; then
                echo "OpenCV version is exactly $minimum_version."
                echo "OpenCV installation complete. "
            else
                echo "OpenCV version is $installed_version, which is higher than $minimum_version."
                manually_opencv_install=false
            fi
        else
            echo "OpenCV version is $installed_version, which is lower than $minimum_version or install opencv by source build. "
            define_list=""
            if [ "$cross_compile_opencv_mode" == false ]; then
                lscpu | grep -i "Vendor ID" | grep -i Intel > /dev/null
                if [ $? -eq 0 ] && [ ! $manually_opencv_install ]; then
                    define_list="$define_list -D WITH_IPP=ON"
                    echo "OpenCV Build With Intel IPP(Integrated Performance Primitives)."
                fi

                libpng-config --version > /dev/null
                if [ $? -eq 0 ]; then
                    define_list="$define_list -D BUILD_PNG=ON -D WITH_PNG=ON"
                else
                    sudo apt-get -y install libpng-dev
                    if [ $? -eq 0 ]; then
                        define_list="$define_list -D BUILD_PNG=ON -D WITH_PNG=ON"
                    fi
                fi
            else
                define_list=" -D BUILD_PNG=ON -D WITH_PNG=ON "
            fi

            if ! test -e $DX_SRC_DIR/util; then 
                mkdir $DX_SRC_DIR/util
            fi
            cd $DX_SRC_DIR/util
            if ! test -e $DX_SRC_DIR/util/opencv.$suggestion_version.zip; then
                wget -O opencv.$suggestion_version.zip https://github.com/opencv/opencv/archive/$suggestion_version.zip 
            fi
            if ! test -e $DX_SRC_DIR/util/opencv_contrib.$suggestion_version.zip; then
                wget -O opencv_contrib.$suggestion_version.zip https://github.com/opencv/opencv_contrib/archive/$suggestion_version.zip
            fi
            echo " unzip opencv & opencv contrib "
            if test -e $DX_SRC_DIR/util/opencv-$suggestion_version/; then
                sudo rm -rf $DX_SRC_DIR/util/opencv-$suggestion_version
            fi
            if test -e $DX_SRC_DIR/util/opencv_contrib-$suggestion_version/; then
                sudo rm -rf $DX_SRC_DIR/util/opencv_contrib-$suggestion_version
            fi

            unzip opencv.$suggestion_version.zip
            unzip opencv_contrib.$suggestion_version.zip
            cd opencv-$suggestion_version
            mkdir build_$target_arch
            cd build_$target_arch
            make clean 
            cmake \
            $toolchain_define \
            $define_list \
            -D BUILD_LIST="imgcodecs,imgproc,core,highgui,videoio" \
	        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-$suggestion_version/modules \
            -D CMAKE_BUILD_TYPE=RELEASE \
            -D WITH_TBB=ON -D WITH_PYTHON=ON -D WITH_QT=OFF -D WITH_GTK=ON \
            -D WITH_V4L=ON -D WITH_FFMPEG=ON \
            -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF \
            -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
            -D BUILD_NEW_PYTHON_SUPPORT=ON \
            -D OPENCV_GENERATE_PKGCONFIG=ON -D WITH_CUDA=OFF ../
            make -j $(($(nproc) / 2))
            
            if [ $? -ne 0 ]; then
                echo "Failed to install OpenCV dependent libraries."
                exit 1
            fi
            
            sudo make install

            if [ $? -ne 0 ]; then
                echo "Failed to install OpenCV dependent libraries."
                exit 1
            fi

            sudo ldconfig
        fi
    fi
}

function install_python() {
    print_colored "--- setup python... ---" "INFO"

    local INSTALL_PY_CMD_ARGS=""

    if [ -n "${PYTHON_VERSION}" ]; then
        INSTALL_PY_CMD_ARGS+=" --python_version=$PYTHON_VERSION"
    fi

    if [ -n "${MIN_PY_VERSION}" ]; then
        INSTALL_PY_CMD_ARGS+=" --min_py_version=$MIN_PY_VERSION"
    fi

    if [ -n "${VENV_PATH}" ]; then
        INSTALL_PY_CMD_ARGS+=" --venv_path=$VENV_PATH"
    fi

    if [ "${VENV_FORCE_REMOVE}" = "y" ]; then
        INSTALL_PY_CMD_ARGS+=" --venv-force-remove"
    fi

    if [ "${VENV_REUSE}" = "y" ]; then
        INSTALL_PY_CMD_ARGS+=" --venv-reuse"
    fi

    # Pass the determined VENV_PATH and new options to install_python_and_venv.sh
    INSTALL_PY_CMD="${DX_APP_PATH}/scripts/install_python_and_venv.sh ${INSTALL_PY_CMD_ARGS}"
    echo "CMD: ${INSTALL_PY_CMD}"
    ${INSTALL_PY_CMD}
    if [ $? -ne 0 ]; then
        print_colored "Python and Virtual environment setup failed. Exiting." "ERROR"
        exit 1
    fi

    print_colored "[OK] Completed to setup python" "INFO"
}

# parse args
[ $# -gt 0 ] && \
while (( $# )); do
    case "$1" in
        --arch)  
            shift
            target_arch=$1
            shift;;       
        --dep) install_dep=true; shift;;        
        --opencv) install_opencv=true; shift;;     
        --opencv-source-build) opencv_source_build=true; shift;;     
        --all) install_opencv=true;install_dep=true; shift;;
        --python_version)
            shift
            PYTHON_VERSION=$1
            shift;;
        --venv_path)
            shift
            VENV_PATH=$1
            shift
            ;;
        -f|--venv-force-remove)
            VENV_FORCE_REMOVE="y"
            shift # past argument
            ;;
        -r|--venv-reuse)
            VENV_REUSE="y"
            shift # past argument
            ;;
        --verbose)
            ENABLE_DEBUG_LOGS=1
            shift # Consume argument
            ;;
        --help) 
            help; 
            exit 0
            ;;
        *)
            help "error"  "Invalid argument: $1"
            exit 1
            ;;
    esac
done

if [ $target_arch == "arm64" ]; then
    target_arch=aarch64
    print_colored_v2 "INFO" " Use arch64 instead of arm64"
    exit 1
fi

install_dep
install_opencv

popd >&2
