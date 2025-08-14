#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}")

# color env settings
source "${DX_APP_PATH}/scripts/color_env.sh"
source "${DX_APP_PATH}/scripts/common_util.sh"

pushd $DX_APP_PATH

print_colored "DX_APP_PATH: $DX_APP_PATH" "INFO"

# Check if bin directory exists and contains files
if [ ! -d "./bin" ] || [ -z "$(ls -A ./bin 2>/dev/null)" ]; then
    print_colored "dx_app is not built. Building dx_app first before running the demo." "INFO"
    ./build.sh
fi

check_valid_dir_or_symlink() {
    local path="$1"
    if [ -d "$path" ] || { [ -L "$path" ] && [ -d "$(readlink -f "$path")" ]; }; then
        return 0
    else
        return 1
    fi
}

if check_valid_dir_or_symlink "./assets/models" && check_valid_dir_or_symlink "./assets/videos"; then
    print_colored "Models and Videos directory already exists. Skipping download." "INFO"
else
    print_colored "Models and Videos not found. Downloading now via setup.sh..." "INFO"
    ./setup.sh --force
fi

WRC=$DX_APP_PATH

echo "0: Object Detection (YOLOv7)"
echo "1: Object Detection (YOLOv8N)"
echo "2: Object Detection (YOLOv9S)"
echo "3: Face Detection (YOLOV5S_Face)"
echo "4: Pose Estimation"
echo "5: Semantic Segmentation"
echo "6: Multi-Channel Object Detection (YOLOv5)"
echo "7: Multi-Model Object Detection (YOLOv5) & Segmentation"

read -t 10 -p "which AI demo do you want to run:(timeout:10s, default:0)" select

case $select in
    0)$WRC/bin/yolo -m assets/models/YoloV7.dxnn -p 4 -v assets/videos/snowboard.mp4 --target_fps 30 -l;;
    1)$WRC/bin/yolo -m assets/models/YoloV8N.dxnn -p 5 -v assets/videos/boat.mp4 --target_fps 30 -l;;
    2)$WRC/bin/yolo -m assets/models/YOLOV9S.dxnn -p 10 -v assets/videos/carrierbag.mp4 --target_fps 30 -l;;
    3)$WRC/bin/yolo -m assets/models/YOLOV5S_Face-1.dxnn -p 7 -v assets/videos/dance-group.mov --target_fps 30 -l;;
    4)$WRC/bin/pose -m assets/models/YOLOV5Pose640_1.dxnn -v assets/videos/dance-solo.mov --target_fps 30 -l;;
    5)$WRC/bin/segmentation -m assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -v assets/videos/blackbox-city-road.mp4 --target_fps 30 -l;;
    6)$WRC/bin/yolo_multi -c example/yolo_multi/yolo_multi_demo.json;;
    7)$WRC/bin/od_segmentation -m0 assets/models/YOLOV5S_6.dxnn -p0 2 -m1 assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -v assets/videos/blackbox-city-road2.mov -l;;
esac

popd