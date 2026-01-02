#!/bin/bash
pushd .
dxapp_dir=`pwd`
if [[ "$dxapp_dir" == *scripts* ]]; then
    cd ..
    dxapp_dir=`pwd`
fi
dxapp_name="yolo_multi"
echo $dxapp_dir

if ! test -e $dxapp_dir/bin/$dxapp_name; then
    ./build.sh --clean
fi

$dxapp_dir/bin/$dxapp_name -m $dxapp_dir/assets/models/YOLOV5S_3.dxnn -c $dxapp_dir/example/yolo_multi/yolo_multi_demo.json

popd