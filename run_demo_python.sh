#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}")

# color env settings
source "${DX_APP_PATH}/scripts/color_env.sh"
source "${DX_APP_PATH}/scripts/common_util.sh"

pushd $DX_APP_PATH

print_colored "DX_APP_PATH: $DX_APP_PATH" "INFO"

# Check if assets exist
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

echo "=== Pure Python Examples (Python Post-Processing) ==="
echo "0: Object Detection Sync (YOLOv5-512)"
echo "1: Object Detection Async (YOLOv7-640)"

echo "=== Pybind11 Examples (Optimized C++ Post-Processing) ==="
echo "2: Face Detection With PPU (SCRFD500M-640)"
echo "3: Pose Estimation With PPU (YOLOv5Pose-640)"
echo "4: Object Detection With PPU (YOLOv5S-512)"
echo "5: Object Detection (YOLOv7-640)"
echo "6: Object Detection (YOLOv8N-640)"
echo "7: Object Detection (YOLOv9S-640)"

# Advanced version with user input handling
prompt="which Python AI demo do you want to run? (default:0): "
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

# Define common paths
PY_TEMPLATE_DIR="templates/python"
TEST_DATA_DIR="example/dx_postprocess"
VIDEO_DIR="assets/videos"

# Video files
VIDEO_SNOWBOARD="${VIDEO_DIR}/snowboard.mp4"
VIDEO_BOAT="${VIDEO_DIR}/boat.mp4"
VIDEO_DANCE_GROUP="${VIDEO_DIR}/dance-group.mov"
VIDEO_DANCE_SOLO="${VIDEO_DIR}/dance-solo.mov"
VIDEO_CARRIERBAG="${VIDEO_DIR}/carrierbag.mp4"

case $select in
    0) python3 ${PY_TEMPLATE_DIR}/yolov5s_example.py ;;
    1) python3 ${PY_TEMPLATE_DIR}/yolo_async.py ;;
    
    2) python3 ${PY_TEMPLATE_DIR}/yolo_pybind_example.py --config_path ${TEST_DATA_DIR}/SCRFD500M_PPU.json --video_path ${VIDEO_DANCE_GROUP} --visualize --run_async ;;
    3) python3 ${PY_TEMPLATE_DIR}/yolo_pybind_example.py --config_path ${TEST_DATA_DIR}/YOLOV5Pose_PPU.json --video_path ${VIDEO_DANCE_SOLO} --visualize --run_async ;;
    4) python3 ${PY_TEMPLATE_DIR}/yolo_pybind_example.py --config_path ${TEST_DATA_DIR}/YOLOV5S_PPU.json --video_path ${VIDEO_SNOWBOARD} --visualize --run_async ;;
    5) python3 ${PY_TEMPLATE_DIR}/yolo_pybind_example.py --config_path ${TEST_DATA_DIR}/YoloV7.json --video_path ${VIDEO_SNOWBOARD} --visualize --run_async ;;
    6) python3 ${PY_TEMPLATE_DIR}/yolo_pybind_example.py --config_path ${TEST_DATA_DIR}/YoloV8N.json --video_path ${VIDEO_BOAT} --visualize --run_async ;;
    7) python3 ${PY_TEMPLATE_DIR}/yolo_pybind_example.py --config_path ${TEST_DATA_DIR}/YOLOV9S.json --video_path ${VIDEO_CARRIERBAG} --visualize --run_async ;;
    *) echo "Invalid selection" ;;
esac

popd
