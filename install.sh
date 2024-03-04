#!/bin/bash
# dependencies install script in host
pushd .
cmd=()
DX_SRC_DIR=$PWD
echo "DX_SRC_DIR the default one $DX_SRC_DIR"
target_arch=$(uname -p)  
install_opencv=false  
install_onnx=false  
install_dep=false  
build_type='Release'  

function help()
{
    echo "./install.sh"
    echo "    --help            show this help"
    echo "    --arch            target CPU architecture : [ x86_64, arm64, riscv64 ]"
    echo "    --dep             install dependencies : cmake, gcc, ninja, etc.."
    echo "    --opencv          (optional) install opencv pkg "
    echo "    --onnxruntime     (optional) install onnxruntime library"
    echo "    --all             install dependencies & opencv pkg & onnxruntime library"
}

function install_dep()
{
    cmake_version=3.14
    if [ "$install_dep" == true ]; then
        echo " Install dependence package tools "
        sudo apt-get -y install build-essential make zlib1g-dev libcurl4-openssl-dev wget tar zip
        if [ "$install_onnx" == true ]; then
            echo " Install CMake version 3.18.0, to build onnxruntime "
            cmake_version=3.18
        fi
        if ! test -e $DX_SRC_DIR/util; then 
            mkdir $DX_SRC_DIR/util
        fi
        cd $DX_SRC_DIR/util
        if ! test -e $DX_SRC_DIR/util/cmake-$cmake_version.0; then
            echo " Install CMake v$$cmake_version.0 "
            wget https://cmake.org/files/v$cmake_version/cmake-$cmake_version.0.tar.gz --no-check-certificate    
            tar xvf cmake-$cmake_version.0.tar.gz
        else
            echo " Already Exist CMake "
        fi
        cd cmake-$cmake_version.0
        ./bootstrap --system-curl
        make -j8
        sudo make install 
        sudo apt install ninja-build
        sudo apt-get -y install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
        sudo apt-get -y install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
    fi
}

function install_opencv()
{
    if [ "$install_opencv" == true ]; then
        toolchain_define="-D CMAKE_INSTALL_PREFIX=/usr/local "
        if [ $(uname -p) != "$target_arch" ]; then
            case "$target_arch" in
              arm64) toolchain_define="-D CMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake -D CMAKE_INSTALL_PREFIX=$DX_SRC_DIR/extern/$target_arch "
              ;;
              riscv64) toolchain_define="-D CMAKE_TOOLCHAIN_FILE=../platforms/linux/riscv64-gnu.toolchain.cmake -D CMAKE_INSTALL_PREFIX=$DX_SRC_DIR/extern/$target_arch "
              ;;
            esac  
            if [ $(uname -p) == "aarch64" ] && [ $target_arch == "arm64" ]; then  
                toolchain_define="-D CMAKE_INSTALL_PREFIX=/usr/local "
            else
                echo " OpenCV Cross Compilation "
            fi
        fi
        echo " Install opencv dependent library "
        sudo apt -y install libopencv-dev python3-opencv libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev \
             libavformat-dev libswscale-dev libdc1394-22-dev libxvidcore-dev \
             libx264-dev libxine2-dev libv4l-dev v4l-utils libgstreamer1.0-dev \
             libgstreamer-plugins-base1.0-dev libgtk-3-dev libfreetype*
        
        if ! test -e $DX_SRC_DIR/util; then 
            mkdir $DX_SRC_DIR/util
        fi
        cd $DX_SRC_DIR/util
        if ! test -e $DX_SRC_DIR/util/opencv.4.5.5.zip; then
            wget -O opencv.4.5.5.zip https://github.com/opencv/opencv/archive/4.5.5.zip 
        fi
        if ! test -e $DX_SRC_DIR/util/opencv_contrib.4.5.5.zip; then
            wget -O opencv_contrib.4.5.5.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip
        fi
        echo " unzip opencv & opencv contrib "
        if test -e $DX_SRC_DIR/util/opencv-4.5.5/; then
            sudo rm -rf $DX_SRC_DIR/util/opencv-4.5.5
        fi
        if test -e $DX_SRC_DIR/util/opencv_contrib-4.5.5/; then
            sudo rm -rf $DX_SRC_DIR/util/opencv_contrib-4.5.5
        fi
        
        unzip opencv.4.5.5.zip
        unzip opencv_contrib.4.5.5.zip
        cd opencv-4.5.5
        mkdir build_$target_arch
        cd build_$target_arch
        make clean 
        cmake \
        $toolchain_define \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.5/modules \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D WITH_TBB=ON -D WITH_IPP=OFF -D WITH_1394=OFF \
        -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF \
        -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON \
        -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
        -D WITH_QT=OFF -D WITH_GTK=ON -D WITH_OPENGL=ON \
        -D WITH_V4L=ON -D WITH_FFMPEG=ON -D WITH_XINE=ON -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON -D WITH_CUDA=OFF -D WITH_FREETYPE=ON ../
        make -j8
        sudo make install
        sudo ldconfig
    fi
}

function install_onnx()
{
    if [ "$install_onnx" == true ]; then
        echo " Install ONNX-Runtime API " 
        if ! test -e $DX_SRC_DIR/util; then 
            mkdir $DX_SRC_DIR/util
        fi
        cd $DX_SRC_DIR/util
        if ! test -e $DX_SRC_DIR/util/onnxruntime-linux-$target_arch-1.13.1.tgz; then
        # get onnxruntime source code release version 1.13.1
            if [ "$target_arch" == "x86_64" ]; then
                wget -O onnxruntime-linux-$target_arch-1.13.1.tgz \
                https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-x64-1.13.1.tgz
            elif [ "$target_arch" == "arm64" ] || [ "$target_arch" == "aarch64" ]; then
                wget -O onnxruntime-linux-$target_arch-1.13.1.tgz \
                https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-aarch64-1.13.1.tgz
            fi
        fi
        sudo tar -xvzf onnxruntime-linux-$target_arch-1.13.1.tgz -C /usr/local --strip-components 1
        sudo ldconfig
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
        --onnxruntime) install_onnx=true; shift;;
        --all) install_onnx=true;install_opencv=true;install_dep=true; shift;;  
        *)       echo "Invalid argument : " $1 ; help; exit 1;;
    esac
done

if [ $target_arch == "aarch64" ]; then
    target_arch=arm64
fi

install_dep
install_opencv
install_onnx

popd
