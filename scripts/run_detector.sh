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

$dxapp_dir/bin/$dxapp_name -c $dxapp_dir/example/yolov5s3_realtime_example.json

popd
