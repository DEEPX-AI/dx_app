#!/bin/bash
dxrt_ver=`cat release.ver`

function help()
{
    echo "./build.sh"
    echo "    --help     show this help"
    echo "    --clean    clean build"
    echo "    --verbose  show build commands"
    echo "    --arch     target CPU architecture : [ x86_64, arm64, riscv64 ]"
}

# cmake command
cmd=()
clean_build=false
verbose=false
target_arch=$(uname -p)
build_mode="Release Build"

[ $# -gt 0 ] && \
while (( $# )); do
    case "$1" in
        --help)  help; exit 0;;
        --clean) clean_build=true; shift;;
        --verbose) verbose=true; shift;;
        --arch)
            shift
            target_arch=$1
            shift;;
        *)       echo "Invalid argument : " $1 ; help; exit 1;;
    esac
done

if [ $target_arch == "aarch64" ]; then
    target_arch=arm64
fi

cmd+=(-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.$target_arch.cmake)

cmd+=(-DCMAKE_VERBOSE_MAKEFILE=$verbose)
cmd+=(-DCMAKE_BUILD_TYPE="release");

cmd+=(-DCMAKE_GENERATOR=Ninja)

build_dir=build_"$target_arch"
out_dir=bin
echo cmake args : ${cmd[@]}
[ $clean_build == "true" ] && rm -rf $build_dir

mkdir $build_dir
mkdir $out_dir
rm -rf $build_dir/release/bin
cd $build_dir
cmake .. ${cmd[@]}
cmake --build . --target install && cd .. && cp $build_dir/release/bin/* $out_dir/
if [ -e $build_dir/release/bin ]; then
    echo Build Done. "($build_mode)"
    echo =================================================
        echo dxrt_ver : $dxrt_ver
        echo clean_build : $clean_build
        echo verbose : $verbose
        echo build_mode : $build_mode
        echo target_arch : $target_arch
    echo =================================================
else
    echo Build Failed.
    exit 1
fi