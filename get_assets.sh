#!/bin/bash
function help()
{
    echo "./build.sh"
    echo "    --help     show this help"
    echo "    --number   enter the regression ID."
}

regrID=3148

[ $# -gt 0 ] && \
while (( $# )); do
    case "$1" in
        --help)  help; exit 0;;
        --number) 
            shift
            regrID=$1
            shift;;
        *)       echo "Invalid argument : " $1 ; help; exit 1;;
    esac
done

mkdir -p assets/$regrID

sudo mkdir -p /mnt/regression_storage
sudo mount -o nolock 192.168.30.201:/do/regression /mnt/regression_storage

for x in \
DeepLabV3PlusMobileNetV2_2 \
EfficientNetB0_4 \
EfficientNetB0_8 \
MobileNetV2_2 \
HyundaiDDRNet_1 \
HyundaiFaceID_1 \
HyundaiPytorchHalfpixel_1 \
HyundaiFaceAlignment \
SCRFD500M_1 \
YOLOV3_1 \
YOLOV4_3 \
YOLOV5Pose640_1 \
YOLOV5S_1 \
YOLOV5S_4 \
YOLOV5S_3 \
YOLOV5S_6 \
YOLOV5X_2 \
YOLOv7_512 \
YoloV7 \
YoloV8N \
;\
do cp -r /mnt/regression_storage/dxnn_regr_data/M1A/$regrID/$x-*/$x.dxnn ./assets/$regrID/ \
&& echo "$x --> DONE";\
done

echo "packing models - $regrID-models.tar.gz"
tar -czvf $regrID-models.tar.gz assets/$regrID

if [ -d "assets/videos" ]; then 
    echo "  The assets/videos folder already exists."
    echo "  Stopping the video download."
    exit 0
else
    mkdir -p assets/videos
    cp -a -n /mnt/regression_storage/lyj/videos/* ./assets/videos/
fi
