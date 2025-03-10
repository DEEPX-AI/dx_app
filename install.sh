#!/bin/bash
# dependencies install script in host
pushd .
cmd=()
DX_SRC_DIR=$PWD
echo "DX_SRC_DIR the default one $DX_SRC_DIR"
target_arch=$(uname -m)  
install_opencv=false  
install_dep=false  
opencv_source_build=false  
build_type='Release'  

function help()
{
    echo "./install.sh"
    echo "    --help                    show this help"
    echo "    --arch                    target CPU architecture : [ x86_64, aarch64, riscv64 ]"
    echo "    --dep                     install dependencies : cmake, gcc, ninja, etc.."
    echo "    --opencv                  (optional) install opencv pkg "
    echo "    --opencv-source-build     (optional) install opencv pkg by source build"
    echo "    --all                     install dependencies & opencv pkg "
}

function compare_version() 
{
    awk -v n1="$1" -v n2="$2" 'BEGIN { if (n1 >= n2) exit 0; else exit 1; }'
}

function install_dep()
{
    cmake_version_required=3.14
    install_cmake=false
    if [ "$install_dep" == true ]; then
        echo " Install dependence package tools "
        sudo apt-get update 
        if [ $? -ne 0 ]; then
            echo "Failed to apt update."
            exit 1
        fi
        sudo apt-get update && apt-get -y install build-essential make zlib1g-dev libcurl4-openssl-dev wget tar zip cmake
        echo ""
        echo " Install python libraries" 
        sudo apt-get -y install python3-dev python3-setuptools python3-pip python3-tk python3-lxml python3-six
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
        sudo apt-get -y install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
    fi
}

function install_opencv()
{
    cross_compile_opencv_mode=false
    if [ "$install_opencv" == true ] || [ "$opencv_source_build" == true ]; then
        toolchain_define="-D CMAKE_INSTALL_PREFIX=/usr/local "
        if [ $(uname -m) != "$target_arch" ]; then
            case "$target_arch" in
              arm64) toolchain_define="-D CMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake -D CMAKE_INSTALL_PREFIX=$DX_SRC_DIR/extern/$target_arch "
              ;;
              aarch64) toolchain_define="-D CMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake -D CMAKE_INSTALL_PREFIX=$DX_SRC_DIR/extern/$target_arch "
              ;;
              riscv64) toolchain_define="-D CMAKE_TOOLCHAIN_FILE=../platforms/linux/riscv64-gnu.toolchain.cmake -D CMAKE_INSTALL_PREFIX=$DX_SRC_DIR/extern/$target_arch "
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
             libtbb-dev libeigen3-dev libx264-dev libv4l-dev v4l-utils libgstreamer1.0-dev \
             libgstreamer-plugins-base1.0-dev libgtk2.0-dev libfreetype-dev

        if [ $? -ne 0 ]; then
            sudo apt-get clean && sudo apt update && sudo apt-get -y upgrade
            sudo apt -y install libgstreamer-plugins-base1.0-dev
        fi

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

[ $# -gt 0 ] && \
while (( $# )); do
    case "$1" in
        --help) help; exit 0;;      
        --arch)  
            shift
            target_arch=$1
            shift;;       
        --dep) install_dep=true; shift;;        
        --opencv) install_opencv=true; shift;;     
        --opencv-source-build) opencv_source_build=true; shift;;     
        --all) install_opencv=true;install_dep=true; shift;;  
        *)       echo "Invalid argument : " $1 ; help; exit 1;;
    esac
done

if [ $target_arch == "arm64" ]; then
    target_arch=aarch64
    echo " Use arch64 instead of arm64"
    exit 1
fi

install_dep
install_opencv

popd
