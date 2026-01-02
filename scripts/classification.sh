#!/bin/bash
pushd .
dxapp_dir=`pwd`
if [[ "$dxapp_dir" == *scripts* ]]; then
    cd ..
    dxapp_dir=`pwd`
fi
dxapp_name="classification"
echo $dxapp_dir

if ! test -e $dxapp_dir/bin/$dxapp_name; then
    ./build.sh --clean
fi

$dxapp_dir/bin/$dxapp_name -m $dxapp_dir/assets/models/EfficientNetB0_4.dxnn -i $dxapp_dir/sample/ILSVRC2012/1.jpeg -l 10

popd