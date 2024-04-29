#!/bin/bash
pushd .
dxapp_dir=`pwd`
if [[ "$dxapp_dir" == *scripts* ]]; then
    cd ..
    dxapp_dir=`pwd`
fi
dxapp_name="run_detector"
echo $dxapp_dir

if ! test -e $dxapp_dir/bin/$dxapp_name; then
    ./build.sh --clean
fi

sudo $dxapp_dir/bin/$dxapp_name -c $dxapp_dir/configs/yolov5s_example.json

popd
