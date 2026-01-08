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

# Check and set LD_LIBRARY_PATH
if [[ ":$LD_LIBRARY_PATH:" != *":$WRC/lib:"* ]]; then
    print_colored "Adding $WRC/lib to LD_LIBRARY_PATH" "INFO"
    export LD_LIBRARY_PATH="$WRC/lib:$LD_LIBRARY_PATH"
else
    print_colored "LD_LIBRARY_PATH already contains $WRC/lib" "INFO"
fi

echo "0: Object Detection (YOLOv7)"
echo "1: Object Detection with PPU (YOLOv7-640)"
echo "2: Object Detection (YOLOv8N)"
echo "3: Object Detection (YOLOv9S)"
echo "4: Object Detection With PPU (YOLOv5S-512)"
echo "5: Face Detection (YOLOV5S_Face)"
echo "6: Face Detection With PPU (SCRFD500M-640)"
echo "7: Pose Estimation"
echo "8: Pose Estimation With PPU (YOLOv5Pose-640)"
echo "9: Semantic Segmentation"
echo "10: Multi-Channel Object Detection (YOLOv5)"
echo "11: Multi-Model Object Detection (YOLOv7) & Segmentation"

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
    0)$WRC/bin/yolov7_async -m assets/models/YoloV7.dxnn -v assets/videos/snowboard.mp4;;
    1)$WRC/bin/yolov7_ppu_async -m assets/models/YoloV7_PPU.dxnn -v assets/videos/snowboard.mp4;;
    2)$WRC/bin/yolov8_async -m assets/models/YoloV8N.dxnn -v assets/videos/boat.mp4;;
    3)$WRC/bin/yolov9_async -m assets/models/YOLOV9S.dxnn -v assets/videos/carrierbag.mp4;;
    4)$WRC/bin/yolov5_ppu_async -m assets/models/YOLOV5S_PPU.dxnn -v assets/videos/boat.mp4;;
    5)$WRC/bin/yolov5face_async -m assets/models/YOLOV5S_Face-1.dxnn -v assets/videos/dance-group.mov;;
    6)$WRC/bin/scrfd_ppu_async -m assets/models/SCRFD500M_PPU.dxnn -v assets/videos/dance-group.mov;;
    7)$WRC/bin/yolov5pose_async -m assets/models/YOLOV5Pose640_1.dxnn -v assets/videos/dance-solo.mov;;
    8)$WRC/bin/yolov5pose_ppu_async -m assets/models/YOLOV5Pose_PPU.dxnn -v assets/videos/dance-solo.mov;;
    9)$WRC/bin/deeplabv3_async -m assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -v assets/videos/blackbox-city-road.mp4;;
    10)$WRC/bin/demo_multi_channel assets/models/YOLOV5S_3.dxnn assets/videos/snowboard.mp4 assets/videos/blackbox-city-road.mp4 assets/videos/boat.mp4 assets/videos/carrierbag.mp4;;
    11)$WRC/bin/yolov7_x_deeplabv3_async -y assets/models/YoloV7.dxnn -d assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -v assets/videos/blackbox-city-road2.mov;;
esac

popd