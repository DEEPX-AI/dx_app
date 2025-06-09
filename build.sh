#!/bin/bash
function help()
{
    echo "./build.sh"
    echo "    --help     show this help"
    echo "    --clean    clean build"
    echo "    --verbose  show build commands"
    echo "    --type     cmake build type : [ Release, Debug, RelWithDebInfo ]"
    echo "    --arch     target CPU architecture : [ x86_64, aarch64, riscv64 ]"
}

# cmake command
cmd=()
clean_build=false
verbose=false
target_arch=$(uname -m)
build_type=release  
build_gtest=false

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
        --test) build_gtest=true; shift;;
        *)       echo "Invalid argument : " $1 ; help; exit -1;;
    esac
done

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
    pip install .
    install_result=$?
    cd ../..
    if [ $install_result -eq 0 ]; then
        echo "dx_postprocess module installed successfully"
        # Verify installation
        python -c "import dx_postprocess; print('dx_postprocess module verification: OK')" 2>/dev/null || echo "-- Warning: dx_postprocess module import failed"
    else
        echo "-- Warning: Failed to install dx_postprocess module"
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