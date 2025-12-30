#!/bin/bash
pushd .
dxapp_dir=`pwd`
if [[ "$dxapp_dir" == *scripts* ]]; then
    cd ..
    dxapp_dir=`pwd`
fi
dxapp_name="pose"
echo $dxapp_dir

if ! test -e $dxapp_dir/bin/$dxapp_name; then
    ./build.sh --clean
fi

$dxapp_dir/bin/$dxapp_name -m $dxapp_dir/assets/models/YOLOV5Pose640_1.dxnn -p 0 -v $dxapp_dir/assets/videos/dance-group2.mov

popd