#!/bin/bash
# =============================================================================
# run_demo.sh - DX-APP Unified Interactive Demo
# =============================================================================
# 3-stage interactive menu: Task → Mode → Input
# Supports both C++ and Python examples in a single script.
#
# Usage:
#   ./run_demo.sh                                # Full interactive
#   ./run_demo.sh --task 0                       # Skip task selection
#   ./run_demo.sh --task 0 --mode 2 --input 1   # Fully non-interactive
#   ./run_demo.sh --help                         # Show help
# =============================================================================

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}")

source "${DX_APP_PATH}/scripts/color_env.sh"
source "${DX_APP_PATH}/scripts/common_util.sh"

# =============================================================================
# Demo Registry (18 entries)
# =============================================================================
DEMO_LABELS=(
    # ── Detection (4) ──
    "Object Detection         (YOLOv7)"
    "Object Detection         (YOLOv11N)"
    "Face Detection           (SCRFD500M)"
    "OBB Detection            (YOLO26N-OBB)"
    # ── Pose & Landmark (3) ──
    "Pose Estimation          (YOLOv8s-Pose)"
    "Hand Landmark            (HandLandmarkLite)"
    "Face Alignment           (3DDFA-V2-MobileNetV1)"
    # ── Segmentation (2) ──
    "Instance Segmentation    (YOLOv8N-Seg)"
    "Semantic Segmentation    (DeepLabV3+MobileNet)"
    # ── Classification (1) ──
    "Classification           (ResNet50)"
    # ── Depth Estimation (1) ──
    "Depth Estimation         (SCDepthV3)"
    # ── Image Restoration (3) ──
    "Image Denoising          (DnCNN-50)"
    "Super Resolution         (ESPCN-X4)"
    "Image Enhancement        (Zero-DCE)"
    # ── Recognition (3) ──
    "Embedding                (ArcFace)"
    "Attribute Recognition    (DeepMAR-ResNet50)"
    "Person Re-ID             (CasViT-T)"
    # ── PPU (1) ──
    "PPU Pipeline             (YOLOv7-PPU)"
)

DEMO_GROUPS=(
    "Detection" "Detection" "Detection" "Detection"
    "Pose & Landmark" "Pose & Landmark" "Pose & Landmark"
    "Segmentation" "Segmentation"
    "Classification"
    "Depth Estimation"
    "Image Restoration" "Image Restoration" "Image Restoration"
    "Recognition" "Recognition" "Recognition"
    "PPU"
)

DEMO_CPP_BASE=(
    yolov7 yolov11n scrfd500m yolo26n_obb
    yolov8s_pose handlandmarklite_1 3ddfa_v2_mobilnetv1_120x120
    yolov8n_seg deeplabv3plusmobilenet
    resnet50
    scdepthv3
    dncnn_50 espcn_x4 zero_dce
    arcface_mobilefacenet deepmar_resnet50 casvit_t
    yolov7_ppu
)

DEMO_PY_DIR=(
    "object_detection/yolov7"
    "object_detection/yolov11n"
    "face_detection/scrfd500m"
    "obb_detection/yolo26n_obb"
    "pose_estimation/yolov8s_pose"
    "hand_landmark/handlandmarklite_1"
    "face_alignment/3ddfa_v2_mobilnetv1_120x120"
    "instance_segmentation/yolov8n_seg"
    "semantic_segmentation/deeplabv3plusmobilenet"
    "classification/resnet50"
    "depth_estimation/scdepthv3"
    "image_denoising/dncnn_50"
    "super_resolution/espcn_x4"
    "image_enhancement/zero_dce"
    "embedding/arcface_mobilefacenet"
    "attribute_recognition/deepmar_resnet50"
    "reid/casvit_t"
    "ppu/yolov7_ppu"
)

DEMO_PY_BASE=(
    yolov7 yolov11n scrfd500m yolo26n_obb
    yolov8s_pose handlandmarklite_1 3ddfa_v2_mobilnetv1_120x120
    yolov8n_seg deeplabv3plusmobilenet
    resnet50
    scdepthv3
    dncnn_50 espcn_x4 zero_dce
    arcface_mobilefacenet deepmar_resnet50 casvit_t
    yolov7_ppu
)

DEMO_MODEL=(
    YoloV7.dxnn YOLOV11N.dxnn SCRFD500M.dxnn yolo26n-obb.dxnn
    yolov8s_pose.dxnn HandLandmarkLite_1.dxnn 3ddfa_v2_mobilnetv1_120x120.dxnn
    yolov8n_seg.dxnn DeepLabV3PlusMobilenet.dxnn
    ResNet50.dxnn
    scdepthv3.dxnn
    DnCNN_50.dxnn ESPCN_X4.dxnn zero_dce.dxnn
    arcface_mobilefacenet.dxnn deepmar_resnet50.dxnn casvit_t.dxnn
    YoloV7_PPU.dxnn
)

DEMO_VIDEO=(
    "assets/videos/snowboard.mp4"
    "assets/videos/boat.mp4"
    "assets/videos/dance-group.mov"
    "assets/videos/dron-citry-road.mov"
    "assets/videos/dance-solo.mov"
    "assets/videos/hand.mp4"
    "assets/videos/face-alignment-closeup.mp4"
    "assets/videos/dogs.mp4"
    "assets/videos/blackbox-city-road.mp4"
    "assets/videos/dogs.mp4"
    "assets/videos/blackbox-city-road.mp4"
    "assets/videos/dance-group.mov"
    "assets/videos/dance-group.mov"
    "assets/videos/dance-group.mov"
    "assets/videos/face-pair-sofa.mp4"
    "assets/videos/person-pair-hallway.mp4"
    "assets/videos/person-pair-hallway.mp4"
    "assets/videos/snowboard.mp4"
)

DEMO_IMAGE=(
    "sample/img/sample_street.jpg"
    "sample/img/sample_street.jpg"
    "sample/img/sample_face.jpg"
    "sample/dota8_test/P0284.png"
    "sample/img/sample_people.jpg"
    "sample/img/sample_hand.jpg"
    "sample/img/sample_face_a1.jpg"
    "sample/img/sample_street.jpg"
    "sample/img/sample_parking.jpg"
    "sample/img/sample_dog.jpg"
    "sample/img/sample_parking.jpg"
    "sample/img/sample_denoising.jpg"
    "sample/img/sample_superresolution.png"
    "sample/img/sample_lowlight.jpg"
    "sample/img/face_pair"
    "sample/img/sample_person_a1.jpg"
    "sample/img/person_pair"
    "sample/img/sample_street.jpg"
)

# "full" = all 6 modes, "no_py_async" = classification only (no async python)
DEMO_PY_ASYNC=(
    full full full full
    full full full
    full full
    no_py_async
    full
    full full full
    full full full
    full
)

# 1 = image only (skip video selection), 0 = both image and video
DEMO_IMAGE_ONLY=(
    0 0 0 0
    0 0 0
    0 0
    0
    0
    0 0 0
    1 1 1
    0
)

DEMO_COUNT=${#DEMO_LABELS[@]}

# =============================================================================
# Functions
# =============================================================================

usage() {
    cat <<EOF
DX-APP Interactive Demo

Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    --task NUM     Pre-select task (0-$((DEMO_COUNT-1)))
    --mode NUM     Pre-select mode (1-6)
    --input NUM    Pre-select input type (1=video, 2=image)
    --show-log     Enable verbose log output (default: quiet)
    --help         Show this help message

Interactive 3-stage menu:
    Stage 1: Select AI task (17 options)
    Stage 2: Select language + execution mode (up to 6 options)
    Stage 3: Select input type (video or image)

Press Enter at any prompt to accept the default selection.
EOF
    exit 0
}

check_valid_dir_or_symlink() {
    local path="$1"
    if [ -d "$path" ] || { [ -L "$path" ] && [ -d "$(readlink -f "$path")" ]; }; then
        return 0
    fi
    return 1
}

print_intro() {
    echo ""
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
    printf "  ${COLOR_BOLD}${COLOR_CYAN}DX-APP Unified Interactive Demo${COLOR_RESET}\n"
    printf "  Datexel NPU Inference Demo  ·  %d AI Tasks available\n" "$DEMO_COUNT"
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
    echo ""
    printf "  Usage:\n"
    printf "    ./run_demo.sh                               Full interactive\n"
    printf "    ./run_demo.sh --task 0 --mode 2 --input 1  Skip all menus\n"
    printf "    ./run_demo.sh --show-log                   Enable verbose logs\n"
    printf "    ./run_demo.sh --help                       Show help\n"
    echo ""
    printf "  ${COLOR_YELLOW}TIP:${COLOR_RESET} To run ${COLOR_BOLD}all 130+ models${COLOR_RESET} (beyond the %d demo tasks),\n" "$DEMO_COUNT"
    printf "       use the DX Model Tool:\n"
    printf "         ${COLOR_GREEN}./scripts/dx_tool.sh run${COLOR_RESET}    ← interactive category/model filter\n"
    printf "         ${COLOR_GREEN}./scripts/dx_tool.sh bench${COLOR_RESET}  ← benchmark with performance report\n"
    echo ""
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
    printf "  Press ${COLOR_BOLD}[Enter]${COLOR_RESET} to start, or ${COLOR_BOLD}Ctrl+C${COLOR_RESET} to exit."
    read -r
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
}

# =============================================================================
# Parse CLI arguments
# =============================================================================
ARG_TASK=""
ARG_MODE=""
ARG_INPUT=""
ARG_SHOW_LOG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)     ARG_TASK="$2"; shift 2 ;;
        --mode)     ARG_MODE="$2"; shift 2 ;;
        --input)    ARG_INPUT="$2"; shift 2 ;;
        --show-log) ARG_SHOW_LOG="1"; shift ;;
        --help)     usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# =============================================================================
# Prerequisites
# =============================================================================
pushd "$DX_APP_PATH" > /dev/null

print_colored "DX_APP_PATH: $DX_APP_PATH" "INFO"

if [ ! -d "./bin" ] || [ -z "$(ls -A ./bin 2>/dev/null)" ]; then
    print_colored "dx_app is not built. Building first..." "INFO"
    ./build.sh
fi

# Ensure videos directory exists
if ! check_valid_dir_or_symlink "./assets/videos"; then
    print_colored "Videos not found. Running setup for videos..." "INFO"
    ./setup_sample_videos.sh --output=./assets/videos
fi

# Ensure all 18 demo models are present; download any missing ones
MODELS_DIR="./assets/models"
if [ -L "$MODELS_DIR" ]; then
    MODELS_REAL=$(readlink -f "$MODELS_DIR")
else
    MODELS_REAL="$MODELS_DIR"
fi
mkdir -p "$MODELS_REAL"
MISSING_DEMO_MODELS=()
for model_file in "${DEMO_MODEL[@]}"; do
    if [ ! -f "${MODELS_REAL}/${model_file}" ]; then
        MISSING_DEMO_MODELS+=("${model_file%.dxnn}")
    fi
done

if [ ${#MISSING_DEMO_MODELS[@]} -gt 0 ]; then
    print_colored "Missing ${#MISSING_DEMO_MODELS[@]} of ${#DEMO_MODEL[@]} demo model(s)." "WARNING"
    print_colored "Missing: ${MISSING_DEMO_MODELS[*]}" "WARNING"
    print_colored "Automatically downloading missing demo models... (no manual setup.sh needed)" "INFO"
    ./setup_sample_models.sh --output="${MODELS_REAL}" --models ${MISSING_DEMO_MODELS[*]}
    if [ $? -ne 0 ]; then
        print_colored "Failed to download demo models." "ERROR"
        print_colored "You can also download manually: ./setup.sh --models ${MISSING_DEMO_MODELS[*]}" "INFO"
        popd > /dev/null
        exit 1
    fi
    print_colored "Demo models ready. Continuing..." "INFO"
else
    print_colored "All ${#DEMO_MODEL[@]} demo models found." "INFO"
fi

WRC="$DX_APP_PATH"

if [[ ":$LD_LIBRARY_PATH:" != *":$WRC/lib:"* ]]; then
    export LD_LIBRARY_PATH="$WRC/lib:$LD_LIBRARY_PATH"
fi

# =============================================================================
# Intro Banner (interactive mode only)
# =============================================================================
if [ -z "$ARG_TASK" ] && [ -z "$ARG_MODE" ] && [ -z "$ARG_INPUT" ] && [ -z "$ARG_SHOW_LOG" ]; then
    print_intro
fi

# =============================================================================
# Stage 1: Task Selection
# =============================================================================
if [ -n "$ARG_TASK" ]; then
    task_sel="$ARG_TASK"
    if ! [[ "$task_sel" =~ ^[0-9]+$ ]] || [ "$task_sel" -ge "$DEMO_COUNT" ]; then
        print_colored "Invalid task: $task_sel" "ERROR"
        popd > /dev/null
        exit 1
    fi
else
    echo ""
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
    printf "  ${COLOR_BOLD}${COLOR_CYAN}[ Stage 1 / 3 ]  Select AI Task${COLOR_RESET}\n"
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"

    prev_group=""
    for ((i=0; i<DEMO_COUNT; i++)); do
        group="${DEMO_GROUPS[$i]}"
        if [ "$group" != "$prev_group" ]; then
            printf "\n  ${COLOR_YELLOW}[ %s ]${COLOR_RESET}\n" "$group"
            prev_group="$group"
        fi
        printf "   %2d: %s\n" "$i" "${DEMO_LABELS[$i]}"
    done

    echo ""
    while true; do
        printf "  Select task [0-%d, default: 0]: " "$((DEMO_COUNT-1))"
        read -r task_sel
        [[ -z "$task_sel" ]] && task_sel="0"
        if [[ "$task_sel" =~ ^[0-9]+$ ]] && [ "$task_sel" -lt "$DEMO_COUNT" ]; then
            break
        fi
        printf "${COLOR_RED}  Invalid: '%s'. Enter a number between 0 and %d.${COLOR_RESET}\n" "$task_sel" "$((DEMO_COUNT-1))"
    done
fi

print_colored "Task: ${DEMO_LABELS[$task_sel]}" "INFO"

# =============================================================================
# Stage 2: Mode Selection
# =============================================================================
MODE_NAMES=()
MODE_KEYS=()

MODE_NAMES+=("cpp_sync")
MODE_KEYS+=("cpp_sync")
MODE_NAMES+=("cpp_async")
MODE_KEYS+=("cpp_async")
MODE_NAMES+=("py_sync")
MODE_KEYS+=("py_sync")

if [ "${DEMO_PY_ASYNC[$task_sel]}" = "full" ]; then
    MODE_NAMES+=("py_async")
    MODE_KEYS+=("py_async")
fi

MODE_NAMES+=("py_sync_cpp_postprocess")
MODE_KEYS+=("py_sync_cpp_postprocess")

if [ "${DEMO_PY_ASYNC[$task_sel]}" = "full" ]; then
    MODE_NAMES+=("py_async_cpp_postprocess")
    MODE_KEYS+=("py_async_cpp_postprocess")
fi

mode_count=${#MODE_NAMES[@]}

if [ -n "$ARG_MODE" ]; then
    mode_sel="$ARG_MODE"
    if ! [[ "$mode_sel" =~ ^[0-9]+$ ]] || [ "$mode_sel" -lt 1 ] || [ "$mode_sel" -gt "$mode_count" ]; then
        print_colored "Invalid mode: $mode_sel" "ERROR"
        popd > /dev/null
        exit 1
    fi
else
    echo ""
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
    printf "  ${COLOR_BOLD}${COLOR_CYAN}[ Stage 2 / 3 ]  Select Execution Mode${COLOR_RESET}\n"
    printf "  Task : %s\n" "${DEMO_LABELS[$task_sel]}"
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
    echo ""
    for ((i=0; i<mode_count; i++)); do
        printf "   %d: %s\n" "$((i+1))" "${MODE_NAMES[$i]}"
    done
    echo ""
    while true; do
        printf "  Select mode [1-%d, default: 1]: " "$mode_count"
        read -r mode_sel
        [[ -z "$mode_sel" ]] && mode_sel="1"
        if [[ "$mode_sel" =~ ^[0-9]+$ ]] && [ "$mode_sel" -ge 1 ] && [ "$mode_sel" -le "$mode_count" ]; then
            break
        fi
        printf "${COLOR_RED}  Invalid: '%s'. Enter a number between 1 and %d.${COLOR_RESET}\n" "$mode_sel" "$mode_count"
    done
fi

selected_mode="${MODE_KEYS[$((mode_sel-1))]}"
print_colored "Mode: $selected_mode" "INFO"

# =============================================================================
# Stage 3: Input Type Selection
# =============================================================================
if [ "${DEMO_IMAGE_ONLY[$task_sel]}" = "1" ]; then
    # Image-only task (e.g., recognition models) — skip video selection
    input_sel="2"
    print_colored "Input: image only (video not applicable for this task)" "INFO"
elif [ -n "$ARG_INPUT" ]; then
    input_sel="$ARG_INPUT"
else
    echo ""
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
    printf "  ${COLOR_BOLD}${COLOR_CYAN}[ Stage 3 / 3 ]  Select Input Type${COLOR_RESET}\n"
    printf "  Task : %s\n" "${DEMO_LABELS[$task_sel]}"
    printf "  Mode : %s\n" "$selected_mode"
    printf "${COLOR_CYAN}%s${COLOR_RESET}\n" "═══════════════════════════════════════════════════════════════"
    echo ""
    printf "   1: video  (%s)\n" "${DEMO_VIDEO[$task_sel]}"
    printf "   2: image  (%s)\n" "${DEMO_IMAGE[$task_sel]}"
    echo ""
    while true; do
        printf "  Select input [1=video, 2=image, default: 1]: "
        read -r input_sel
        [[ -z "$input_sel" ]] && input_sel="1"
        if [ "$input_sel" = "1" ] || [ "$input_sel" = "2" ]; then
            break
        fi
        printf "${COLOR_RED}  Invalid: '%s'. Enter 1 (video) or 2 (image).${COLOR_RESET}\n" "$input_sel"
    done
fi

if [ "$input_sel" = "2" ]; then
    input_type="image"
    input_file="${DEMO_IMAGE[$task_sel]}"
else
    input_type="video"
    input_file="${DEMO_VIDEO[$task_sel]}"
fi

print_colored "Input: $input_type ($input_file)" "INFO"

# =============================================================================
# Show-Log Selection (interactive mode only)
# =============================================================================
if [ -z "$ARG_SHOW_LOG" ] && [ -z "$ARG_TASK" ] && [ -z "$ARG_MODE" ] && [ -z "$ARG_INPUT" ]; then
    echo ""
    printf "  Enable verbose log output? [y/N, default: N]: "
    read -r _show_log_sel
    if [ "$_show_log_sel" = "y" ] || [ "$_show_log_sel" = "Y" ]; then
        ARG_SHOW_LOG="1"
    fi
fi

# =============================================================================
# Build and Execute Command
# =============================================================================
model_path="assets/models/${DEMO_MODEL[$task_sel]}"
cpp_base="${DEMO_CPP_BASE[$task_sel]}"
py_dir="${DEMO_PY_DIR[$task_sel]}"
py_base="${DEMO_PY_BASE[$task_sel]}"


case "$selected_mode" in
    cpp_sync)
        CMD="$WRC/bin/${cpp_base}_sync -m $model_path"
        ;;
    cpp_async)
        CMD="$WRC/bin/${cpp_base}_async -m $model_path"
        ;;
    py_sync)
        CMD="python3 $WRC/src/python_example/${py_dir}/${py_base}_sync.py --model $model_path"
        ;;
    py_async)
        CMD="python3 $WRC/src/python_example/${py_dir}/${py_base}_async.py --model $model_path"
        ;;
    py_sync_cpp_postprocess)
        CMD="python3 $WRC/src/python_example/${py_dir}/${py_base}_sync_cpp_postprocess.py --model $model_path"
        ;;
    py_async_cpp_postprocess)
        CMD="python3 $WRC/src/python_example/${py_dir}/${py_base}_async_cpp_postprocess.py --model $model_path"
        ;;
esac

# Append verbose flag if show-log is set
if [ -n "$ARG_SHOW_LOG" ]; then
    case "$selected_mode" in
        py_*)  CMD="$CMD --verbose" ;;
        cpp_*) CMD="$CMD --show-log" ;;
    esac
fi

case "$selected_mode" in
    cpp_*)
        if [ "$input_type" = "video" ]; then
            CMD="$CMD -v $input_file"
        else
            CMD="$CMD -i $input_file"
        fi
        ;;
    py_*)
        if [ "$input_type" = "video" ]; then
            CMD="$CMD --video $input_file"
        else
            CMD="$CMD --image $input_file"
        fi
        ;;
esac

# --- Python dependency check (only for Python modes) ---
case "$selected_mode" in
    py_*)
        _missing_deps=()
        python3 -c "import cv2" 2>/dev/null || _missing_deps+=("opencv-python")
        python3 -c "import numpy" 2>/dev/null || _missing_deps+=("numpy")
        if [ ${#_missing_deps[@]} -gt 0 ]; then
            print_colored "Python dependency missing: ${_missing_deps[*]}" "ERROR"
            print_colored "Install with:  pip3 install -r ${DX_APP_PATH}/requirements.txt" "INFO"
            print_colored "Or run:        ./install.sh --dep" "INFO"
            popd > /dev/null
            exit 1
        fi
        ;;
esac

echo ""
printf "${COLOR_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${COLOR_RESET}\n"
printf "  ${COLOR_BOLD}Task  :${COLOR_RESET} %s\n" "${DEMO_LABELS[$task_sel]}"
printf "  ${COLOR_BOLD}Mode  :${COLOR_RESET} %s\n" "$selected_mode"
printf "  ${COLOR_BOLD}Input :${COLOR_RESET} %s (%s)\n" "$input_type" "$input_file"
printf "  ${COLOR_BOLD}Cmd   :${COLOR_RESET} %s\n" "$CMD"
printf "${COLOR_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${COLOR_RESET}\n"
echo ""

eval "$CMD"

popd > /dev/null