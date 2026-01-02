#!/bin/bash
pushd .
dxapp_dir=`pwd`
if [[ "$dxapp_dir" == *scripts* ]]; then
    cd ..
    dxapp_dir=`pwd`
fi
dxapp_name="od_segmentation"
echo $dxapp_dir

if ! test -e $dxapp_dir/bin/$dxapp_name; then
    ./build.sh --clean
fi

$dxapp_dir/bin/$dxapp_name -m0 $dxapp_dir/assets/models/YOLOV5S_3.dxnn -p0 1 -m1 $dxapp_dir/assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -p1 0 -v assets/videos/blackbox-city-road.mp4

popd