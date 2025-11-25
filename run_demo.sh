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
echo "3: Object Detection With PPU (YOLOv5S-512)"
echo "4: Face Detection (YOLOV5S_Face)"
echo "5: Face Detection With PPU (SCRFD500M-640)"
echo "6: Pose Estimation"
echo "7: Pose Estimation With PPU (YOLOv5Pose-640)"
echo "8: Semantic Segmentation"
echo "9: Multi-Channel Object Detection (YOLOv5)"
echo "10: Multi-Channel Object Detection With PPU (YOLOv5-512)"
echo "11: Multi-Model Object Detection (YOLOv5) & Segmentation"

# Advanced version with user input handling
prompt="which AI demo do you want to run? (default:0): "
printf "%s" "$prompt"

for ((i=20; i>0; i--)); do
    # Check for user input (non-blocking)
    read -t 0.1 -n 1 input 2>/dev/null
    if [ $? -eq 0 ]; then
        # User started typing
        read -r rest_input
        select="$input$rest_input"
        break
    fi
    
    # Update countdown
    printf "\r%s(%ds) \033[K" "$prompt" "$i"
    sleep 0.9  # Compensate for the 0.1s read timeout
done

# If no input, set default
if [ -z "$select" ]; then
    printf "\r%s(timeout) \033[K\n" "$prompt"
    select=0
    echo "Using default: 0"
fi

case $select in
    0)$WRC/bin/yolo -m assets/models/YoloV7.dxnn -p 4 -v assets/videos/snowboard.mp4 -l;;
    1)$WRC/bin/yolo -m assets/models/YoloV8N.dxnn -p 5 -v assets/videos/boat.mp4 -l;;
    2)$WRC/bin/yolo -m assets/models/YOLOV9S.dxnn -p 10 -v assets/videos/carrierbag.mp4 -l;;
    3)$WRC/bin/yolo -m assets/models/YOLOV5S_PPU.dxnn -p 11 -v assets/videos/boat.mp4 -l;;
    4)$WRC/bin/yolo -m assets/models/YOLOV5S_Face-1.dxnn -p 7 -v assets/videos/dance-group.mov -l;;
    5)$WRC/bin/yolo -m assets/models/SCRFD500M_PPU.dxnn -p 12 -v assets/videos/dance-group.mov -l;;
    6)$WRC/bin/pose -m assets/models/YOLOV5Pose640_1.dxnn -v assets/videos/dance-solo.mov -l;;
    7)$WRC/bin/pose -m assets/models/YOLOV5Pose_PPU.dxnn -p 1 -v assets/videos/dance-solo.mov -l;;
    8)$WRC/bin/segmentation -m assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -v assets/videos/blackbox-city-road.mp4 -l;;
    9)$WRC/bin/yolo_multi -c example/yolo_multi/yolo_multi_demo.json;;
    10)$WRC/bin/yolo_multi -c example/yolo_multi/ppu_yolo_multi_demo.json;;
    11)$WRC/bin/od_segmentation -m0 assets/models/YOLOV5S_6.dxnn -p0 2 -m1 assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -v assets/videos/blackbox-city-road2.mov -l;;
esac

popd