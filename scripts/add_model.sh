#!/bin/bash
#
# add_model.sh - Copy-from-reference model package generator (C++/Python integrated)
#
# Generates a new model package by COPYING from an existing working reference,
# replacing only names. Supports C++, Python, or both.
#
# Usage:
#   ./add_model.sh <model_name> <task_type> [options]
#
# Task Types:
#   detection               Object Detection (YOLOv5, YOLOv8, etc.)
#   classification          Image Classification (EfficientNet, ResNet, etc.)
#   semantic_segmentation   Semantic Segmentation (DeepLabV3, BiseNet, etc.)
#   instance_segmentation   Instance Segmentation (YOLOv8-seg, YOLOv5-seg)
#   pose                    Pose Estimation (YOLOv5-pose, YOLOv8-pose)
#   face_detection          Face Detection with landmarks (SCRFD, YOLOv5Face)
#   obb                     Oriented Bounding Box (YOLOv26-obb)
#   depth_estimation        Depth Estimation (FastDepth)
#   image_denoising          Image Denoising (DnCNN)
#   super_resolution         Super Resolution (ESPCN)
#   image_enhancement        Image Enhancement (Zero-DCE)
#   ppu                     PPU Pipeline (YOLOv5-ppu, SCRFD-ppu, etc.)
#
# Options:
#   --lang <cpp|py|both>    Language (default: both)
#   --category <name>       Parent folder (default: based on task type)
#   --sync-only             Generate only sync (no async)
#   --postprocessor <type>  yolov5, yolov8, yolox, scrfd, efficientnet, etc.
#   --base-model <name>     Copy from a specific existing model directory
#   --verify                Build and run inference verification after generation
#   --model <path>          Model .dxnn file for --verify
#   --video <path>          Video file for --verify
#   --image <path>          Image file for --verify (default: sample/img/sample_kitchen.jpg)
#   --git-push              After successful --verify, git add/commit/push (asks confirmation)
#   --yes                   Skip confirmation prompt for --git-push (auto-yes)
#   --no-video              Skip video verification when used with --verify (image only)
#   --auto-add [MANIFEST]   Batch-add models from a manifest JSON (non-interactive)
#
# Examples:
#   ./add_model.sh yolov30 detection --postprocessor yolov8
#   ./add_model.sh yolov30 detection --lang cpp --postprocessor yolov8
#   ./add_model.sh yolov30 detection --lang py --base-model yolov26
#   ./add_model.sh new_yolox detection --postprocessor yolox --verify --model model.dxnn
#   ./add_model.sh resnet classification --postprocessor efficientnet --lang both

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DX_APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_SRC_DIR="$DX_APP_ROOT/src/cpp_example"
PY_SRC_DIR="$DX_APP_ROOT/src/python_example"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_file()    { echo -e "${CYAN}  → $1${NC}"; }

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^#//;s/^ //'
    exit 0
}

# ============================================================================
# Parse arguments
# ============================================================================
MODEL_NAME=""
TASK_TYPE=""
LANG_MODE="both"
CATEGORY=""
SYNC_ONLY=false
POSTPROCESSOR=""
BASE_MODEL=""
VERIFY=false
VERIFY_MODEL=""
VERIFY_VIDEO=""
VERIFY_IMAGE=""
NO_VIDEO=false
GIT_PUSH=false
AUTO_YES=false
AUTO_ADD=false
AUTO_ADD_MANIFEST=""

# Help
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,/^$/p' "$0" | sed 's/^#//;s/^ //'
    exit 0
fi

# --auto-add mode does not require positional args
if [ $# -ge 1 ] && [ "$1" = "--auto-add" ]; then
    AUTO_ADD=true
    shift
    if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
        AUTO_ADD_MANIFEST="$1"
        shift
    fi
elif [ $# -lt 2 ]; then
    usage
else
    MODEL_NAME="$1"
    TASK_TYPE="$2"
    shift 2
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --lang)           LANG_MODE="$2";      shift 2 ;;
        --category)       CATEGORY="$2";       shift 2 ;;
        --sync-only)      SYNC_ONLY=true;      shift ;;
        --postprocessor)  POSTPROCESSOR="$2";   shift 2 ;;
        --base-model)     BASE_MODEL="$2";      shift 2 ;;
        --verify)         VERIFY=true;          shift ;;
        --model)          VERIFY_MODEL="$2";    shift 2 ;;
        --video)          VERIFY_VIDEO="$2";    shift 2 ;;
        --image)          VERIFY_IMAGE="$2";    shift 2 ;;
        --no-video)       NO_VIDEO=true;         shift ;;
        --git-push)       GIT_PUSH=true;        shift ;;
        --yes)            AUTO_YES=true;        shift ;;
        *)                print_error "Unknown option: $1"; usage ;;
    esac
done

case "$LANG_MODE" in
    cpp)  LANGS=("cpp") ;;
    py)   LANGS=("py") ;;
    both) LANGS=("cpp" "py") ;;
    *)    print_error "Invalid --lang: $LANG_MODE (cpp|py|both)"; exit 1 ;;
esac

# ============================================================================
# PascalCase conversion (C++ class names)
# ============================================================================
to_pascal_case() {
    local input="$1"
    echo "$input" | sed -E '
        s/yolov([0-9]+)/YOLOv\1/g
        s/yolox/YOLOX/g
        s/scrfd/SCRFD/g
        s/ssd/SSD/g
        s/deeplabv([0-9]+)/DeepLabv\1/g
        s/efficientnet/EfficientNet/g
        s/resnet/ResNet/g
        s/nanodet/NanoDet/g
        s/damoyolo/DamoYOLO/g
        s/bisenetv([0-9]+)/BiseNetV\1/g
        s/bisenet/BiseNet/g
        s/fastdepth/FastDepth/g
        s/dncnn/DnCNN/g
        s/^([a-z])/\u\1/g
    '
}

# Python class name (capitalize first letter; prefix 'N' if starts with digit)
to_py_class() {
    local input="$1"
    # Python identifiers cannot start with a digit
    if [[ "$input" =~ ^[0-9] ]]; then
        echo "N${input}" | sed -E 's/^([a-z])/\u\1/'
    else
        echo "$input" | sed -E 's/^([a-z])/\u\1/'
    fi
}

# Python-safe module name (prefix 'n_' if starts with digit)
to_py_safe_module() {
    local input="$1"
    if [[ "$input" =~ ^[0-9] ]]; then
        echo "n_${input}"
    else
        echo "$input"
    fi
}

# ============================================================================
# Reference mapping: --postprocessor → reference directory & class names
# REF_DIR, REF_MODEL: shared between C++ and Python
# CPP_FACTORY_CLASS, CPP_MODEL_CLASS: C++ specific
# PY_FACTORY_CLASS: Python specific
# ============================================================================
get_reference_info() {
    local pp="$1"
    case "$pp" in
        # Detection: v5-family
        yolov5|yolov3|yolov4)
            REF_DIR="object_detection/yolov5n"; REF_MODEL="yolov5n"
            CPP_FACTORY_CLASS="YOLOv5Factory"; CPP_MODEL_CLASS="YOLOv5"
            PY_FACTORY_CLASS="Yolov5Factory" ;;
        yolov7)
            REF_DIR="object_detection/yolov7"; REF_MODEL="yolov7"
            CPP_FACTORY_CLASS="YOLOv7Factory"; CPP_MODEL_CLASS="YOLOv7"
            PY_FACTORY_CLASS="Yolov7Factory" ;;
        yolox)
            REF_DIR="object_detection/yoloxs"; REF_MODEL="yoloxs"
            CPP_FACTORY_CLASS="YOLOXFactory"; CPP_MODEL_CLASS="YOLOX"
            PY_FACTORY_CLASS="YoloxFactory" ;;

        # Detection: v8-family
        yolov8)
            REF_DIR="object_detection/yolov8n"; REF_MODEL="yolov8n"
            CPP_FACTORY_CLASS="YOLOv8Factory"; CPP_MODEL_CLASS="YOLOv8"
            PY_FACTORY_CLASS="Yolov8Factory" ;;
        yolov9)
            REF_DIR="object_detection/yolov9t"; REF_MODEL="yolov9t"
            CPP_FACTORY_CLASS="YOLOv9tFactory"; CPP_MODEL_CLASS="YOLOv9t"
            PY_FACTORY_CLASS="Yolov9tFactory" ;;
        yolov10)
            REF_DIR="object_detection/yolov10n"; REF_MODEL="yolov10n"
            CPP_FACTORY_CLASS="YOLOv10Factory"; CPP_MODEL_CLASS="YOLOv10"
            PY_FACTORY_CLASS="Yolov10Factory" ;;
        yolov11)
            REF_DIR="object_detection/yolov11n"; REF_MODEL="yolov11n"
            CPP_FACTORY_CLASS="YOLOv11Factory"; CPP_MODEL_CLASS="YOLOv11"
            PY_FACTORY_CLASS="Yolov11Factory" ;;
        yolov12)
            REF_DIR="object_detection/yolov12n"; REF_MODEL="yolov12n"
            CPP_FACTORY_CLASS="YOLOv12nFactory"; CPP_MODEL_CLASS="YOLOv12n"
            PY_FACTORY_CLASS="Yolov12nFactory" ;;
        yolov26)
            REF_DIR="object_detection/yolo26n"; REF_MODEL="yolo26n"
            CPP_FACTORY_CLASS="YOLOv26Factory"; CPP_MODEL_CLASS="YOLOv26"
            PY_FACTORY_CLASS="Yolov26Factory" ;;

        # Detection: special families
        ssd|ssdmv1|ssdmv2lite)
            REF_DIR="object_detection/ssdmv1"; REF_MODEL="ssdmv1"
            CPP_FACTORY_CLASS="SSDMV1Factory"; CPP_MODEL_CLASS="SSDMV1"
            PY_FACTORY_CLASS="Ssdmv1Factory" ;;
        nanodet)
            REF_DIR="object_detection/nanodet_repvgg"; REF_MODEL="nanodet_repvgg"
            CPP_FACTORY_CLASS="NanoDetFactory"; CPP_MODEL_CLASS="NanoDet"
            PY_FACTORY_CLASS="NanodetFactory" ;;
        damoyolo)
            REF_DIR="object_detection/damoyolot"; REF_MODEL="damoyolot"
            CPP_FACTORY_CLASS="DamoYOLOtFactory"; CPP_MODEL_CLASS="DamoYOLOt"
            PY_FACTORY_CLASS="DamoyolotFactory" ;;

        # Face detection
        scrfd)
            REF_DIR="face_detection/scrfd500m"; REF_MODEL="scrfd500m"
            CPP_FACTORY_CLASS="SCRFDFactory"; CPP_MODEL_CLASS="SCRFD"
            PY_FACTORY_CLASS="ScrfdFactory" ;;
        yolov5face|yolov7face)
            REF_DIR="face_detection/yolov5s_face"; REF_MODEL="yolov5s_face"
            CPP_FACTORY_CLASS="YOLOv5FaceFactory"; CPP_MODEL_CLASS="YOLOv5Face"
            PY_FACTORY_CLASS="Yolov5faceFactory" ;;

        # Pose
        yolov5pose)
            REF_DIR="pose_estimation/yolov5pose"; REF_MODEL="yolov5pose"
            CPP_FACTORY_CLASS="YOLOv5PoseFactory"; CPP_MODEL_CLASS="YOLOv5Pose"
            PY_FACTORY_CLASS="Yolov5poseFactory" ;;
        yolov8pose|yolov26pose)
            REF_DIR="pose_estimation/yolov8s_pose"; REF_MODEL="yolov8s_pose"
            CPP_FACTORY_CLASS="YOLOv8PoseFactory"; CPP_MODEL_CLASS="YOLOv8Pose"
            PY_FACTORY_CLASS="Yolov8poseFactory" ;;

        # OBB
        yolov26obb)
            REF_DIR="obb_detection/yolo26n_obb"; REF_MODEL="yolo26n_obb"
            CPP_FACTORY_CLASS="Yolo26n_obbFactory"; CPP_MODEL_CLASS="Yolo26n_obb"
            PY_FACTORY_CLASS="Yolo26n_obbFactory" ;;

        # Classification
        efficientnet|yolov26cls)
            REF_DIR="classification/efficientnet_lite0"; REF_MODEL="efficientnet_lite0"
            CPP_FACTORY_CLASS="EfficientNetFactory"; CPP_MODEL_CLASS="EfficientNet"
            PY_FACTORY_CLASS="EfficientnetFactory" ;;

        # Semantic segmentation
        deeplabv3|deeplab_v3*)
            REF_DIR="semantic_segmentation/deeplabv3plusmobilenet"; REF_MODEL="deeplabv3plusmobilenet"
            CPP_FACTORY_CLASS="DeepLabv3Factory"; CPP_MODEL_CLASS="DeepLabv3"
            PY_FACTORY_CLASS="Deeplabv3Factory" ;;
        bisenetv1|bisenetv2)
            REF_DIR="semantic_segmentation/bisenetv1"; REF_MODEL="bisenetv1"
            CPP_FACTORY_CLASS="BiseNetV1Factory"; CPP_MODEL_CLASS="BiseNetV1"
            PY_FACTORY_CLASS="Bisenetv1Factory" ;;

        # Instance segmentation
        yolov8seg|yolov26seg)
            REF_DIR="instance_segmentation/yolov8n_seg"; REF_MODEL="yolov8n_seg"
            CPP_FACTORY_CLASS="YOLOv8SegFactory"; CPP_MODEL_CLASS="YOLOv8Seg"
            PY_FACTORY_CLASS="Yolov8segFactory" ;;
        yolov5seg)
            REF_DIR="instance_segmentation/yolov5n_seg"; REF_MODEL="yolov5n_seg"
            CPP_FACTORY_CLASS="YOLOv5SegFactory"; CPP_MODEL_CLASS="YOLOv5Seg"
            PY_FACTORY_CLASS="Yolov5segFactory" ;;

        # Depth
        fastdepth)
            REF_DIR="depth_estimation/fastdepth_1"; REF_MODEL="fastdepth_1"
            CPP_FACTORY_CLASS="FastDepth_1Factory"; CPP_MODEL_CLASS="FastDepth_1"
            PY_FACTORY_CLASS="Fastdepth_1Factory" ;;

        # Image denoising
        dncnn)
            REF_DIR="image_denoising/dncnn_15"; REF_MODEL="dncnn_15"
            CPP_FACTORY_CLASS="DnCNN_15Factory"; CPP_MODEL_CLASS="DnCNN_15"
            PY_FACTORY_CLASS="Dncnn_15Factory" ;;

        # PPU
        yolov5_ppu)
            REF_DIR="ppu/yolov5s_ppu"; REF_MODEL="yolov5s_ppu"
            CPP_FACTORY_CLASS="YOLOv5s_ppuFactory"; CPP_MODEL_CLASS="YOLOv5s_ppu"
            PY_FACTORY_CLASS="Yolov5s_ppuFactory" ;;
        yolov7_ppu)
            REF_DIR="ppu/yolov7_ppu"; REF_MODEL="yolov7_ppu"
            CPP_FACTORY_CLASS="YOLOv7PPUFactory"; CPP_MODEL_CLASS="YOLOv7PPU"
            PY_FACTORY_CLASS="Yolov7PpuFactory" ;;
        yolov5pose_ppu)
            REF_DIR="ppu/yolov5pose_ppu"; REF_MODEL="yolov5pose_ppu"
            CPP_FACTORY_CLASS="YOLOv5PosePPUFactory"; CPP_MODEL_CLASS="YOLOv5PosePPU"
            PY_FACTORY_CLASS="Yolov5posePpuFactory" ;;

        # Extended: proper reference models for specialized architectures
        centernet)
            REF_DIR="object_detection/centernet_resnet18"; REF_MODEL="centernet_resnet18"
            CPP_FACTORY_CLASS="CenterNetFactory"; CPP_MODEL_CLASS="CenterNet"
            PY_FACTORY_CLASS="Centernet_resnet18Factory" ;;
        retinaface)
            REF_DIR="face_detection/retinaface_mobilenet0_25_640"; REF_MODEL="retinaface_mobilenet0_25_640"
            CPP_FACTORY_CLASS="RetinaFaceFactory"; CPP_MODEL_CLASS="RetinaFace"
            PY_FACTORY_CLASS="Retinaface_mobilenet0_25_640Factory" ;;
        yolact)
            REF_DIR="instance_segmentation/yolact_regnetx_800mf"; REF_MODEL="yolact_regnetx_800mf"
            CPP_FACTORY_CLASS="YOLACTFactory"; CPP_MODEL_CLASS="YOLACT"
            PY_FACTORY_CLASS="Yolact_regnetx_800mfFactory" ;;
        espcn)
            REF_DIR="super_resolution/espcn_x4"; REF_MODEL="espcn_x4"
            CPP_FACTORY_CLASS="ESPCNFactory"; CPP_MODEL_CLASS="ESPCN"
            PY_FACTORY_CLASS="Espcn_x4Factory" ;;
        segformer)
            REF_DIR="semantic_segmentation/segformer_b0_512x1024"; REF_MODEL="segformer_b0_512x1024"
            CPP_FACTORY_CLASS="SegFormerFactory"; CPP_MODEL_CLASS="SegFormer"
            PY_FACTORY_CLASS="Segformer_b0_512x1024Factory" ;;
        obb)
            REF_DIR="obb_detection/yolo26n_obb"; REF_MODEL="yolo26n_obb"
            CPP_FACTORY_CLASS="Yolo26n_obbFactory"; CPP_MODEL_CLASS="Yolo26n_obb"
            PY_FACTORY_CLASS="Yolo26n_obbFactory" ;;
        zero_dce)
            REF_DIR="image_enhancement/zero_dce"; REF_MODEL="zero_dce"
            CPP_FACTORY_CLASS="ZeroDCEFactory"; CPP_MODEL_CLASS="ZeroDCE"
            PY_FACTORY_CLASS="Zero_dceFactory" ;;
        clip_image)
            REF_DIR="embedding/clip_resnet50_image_encoder_224x224"; REF_MODEL="clip_resnet50_image_encoder_224x224"
            CPP_FACTORY_CLASS="CLIPImageFactory"; CPP_MODEL_CLASS="CLIPImage"
            PY_FACTORY_CLASS="Clip_resnet50_image_encoder_224x224Factory" ;;
        clip_text)
            REF_DIR="embedding/clip_resnet50_text_encoder_77x512"; REF_MODEL="clip_resnet50_text_encoder_77x512"
            CPP_FACTORY_CLASS="CLIPTextFactory"; CPP_MODEL_CLASS="CLIPText"
            PY_FACTORY_CLASS="Clip_resnet50_text_encoder_77x512Factory" ;;
        arcface)
            REF_DIR="embedding/arcface_mobilefacenet"; REF_MODEL="arcface_mobilefacenet"
            CPP_FACTORY_CLASS="ArcFaceFactory"; CPP_MODEL_CLASS="ArcFace"
            PY_FACTORY_CLASS="Arcface_mobilefacenetFactory" ;;

        # Hand landmark (MediaPipe)
        hand_landmark|hand_landmark_lite)
            REF_DIR="hand_landmark/handlandmarklite_1"; REF_MODEL="handlandmarklite_1"
            CPP_FACTORY_CLASS="Handlandmarklite_1Factory"; CPP_MODEL_CLASS="Handlandmarklite_1"
            PY_FACTORY_CLASS="Handlandmarklite_1Factory" ;;

        *)
            print_error "Unknown postprocessor: $pp"
            print_error "Available: yolov5 yolov7 yolov8 yolov9 yolov10 yolov11 yolov12 yolov26"
            print_error "           yolox ssd nanodet damoyolo scrfd yolov5face yolov7face"
            print_error "           yolov5pose yolov8pose yolov26pose yolov26obb"
            print_error "           efficientnet yolov26cls deeplabv3 bisenetv1 bisenetv2"
            print_error "           yolov8seg yolov5seg fastdepth dncnn"
            print_error "           yolov5_ppu yolov7_ppu yolov5pose_ppu"
            print_error "           centernet retinaface"
            print_error "           yolact espcn segformer obb"
            print_error "           zero_dce clip_image clip_text arcface"
            print_error "           hand_landmark"
            exit 1
            ;;
    esac

    # ── Dynamic override: extract actual class names from reference files ──
    # Prevents sed-replacement failures when hardcoded values above don't
    # match what's actually written in the reference source files.
    local _ref_hpp="${CPP_SRC_DIR}/${REF_DIR}/factory/${REF_MODEL}_factory.hpp"
    if [ ! -f "$_ref_hpp" ]; then
        _ref_hpp=$(compgen -G "${CPP_SRC_DIR}/${REF_DIR}/factory/*_factory.hpp" 2>/dev/null | head -1)
    fi
    if [ -f "$_ref_hpp" ]; then
        local _cls; _cls=$(grep -oP 'class \K\w+Factory' "$_ref_hpp" | head -1)
        local _name; _name=$(grep 'getModelName' "$_ref_hpp" | grep -oP 'return "\K[^"]+' | head -1)
        [ -n "$_cls" ]  && CPP_FACTORY_CLASS="$_cls"
        [ -n "$_name" ] && CPP_MODEL_CLASS="$_name"
    fi

    local _ref_py="${PY_SRC_DIR}/${REF_DIR}/factory/${REF_MODEL}_factory.py"
    if [ ! -f "$_ref_py" ]; then
        _ref_py=$(compgen -G "${PY_SRC_DIR}/${REF_DIR}/factory/*_factory.py" 2>/dev/null | head -1)
    fi
    if [ -f "$_ref_py" ]; then
        local _py_cls; _py_cls=$(grep -oP 'class \K\w+Factory' "$_ref_py" | head -1)
        [ -n "$_py_cls" ] && PY_FACTORY_CLASS="$_py_cls"
    fi
}

# ============================================================================
# Set defaults based on task type  (skipped in --auto-add mode)
# ============================================================================
if [ "$AUTO_ADD" = false ]; then
case "$TASK_TYPE" in
    detection|object_detection) [ -z "$CATEGORY" ] && CATEGORY="object_detection"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="yolov8" ;;
    classification)       [ -z "$CATEGORY" ] && CATEGORY="classification"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="efficientnet" ;;
    semantic_segmentation) [ -z "$CATEGORY" ] && CATEGORY="semantic_segmentation"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="deeplabv3" ;;
    instance_segmentation) [ -z "$CATEGORY" ] && CATEGORY="instance_segmentation"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="yolov8seg" ;;
    pose|pose_estimation) [ -z "$CATEGORY" ] && CATEGORY="pose_estimation"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="yolov5pose" ;;
    face_detection)       [ -z "$CATEGORY" ] && CATEGORY="face_detection"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="scrfd" ;;
    obb|obb_detection)    [ -z "$CATEGORY" ] && CATEGORY="obb_detection"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="yolov26obb"
                          SYNC_ONLY=true ;;
    depth_estimation)     [ -z "$CATEGORY" ] && CATEGORY="depth_estimation"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="fastdepth" ;;
    image_denoising)      [ -z "$CATEGORY" ] && CATEGORY="image_denoising"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="dncnn" ;;
    super_resolution)     [ -z "$CATEGORY" ] && CATEGORY="super_resolution"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="espcn" ;;
    image_enhancement)    [ -z "$CATEGORY" ] && CATEGORY="image_enhancement"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="zero_dce" ;;
    ppu)                  [ -z "$CATEGORY" ] && CATEGORY="ppu"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="yolov5_ppu" ;;
    embedding)            [ -z "$CATEGORY" ] && CATEGORY="embedding"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="clip_image" ;;
    hand_landmark)        [ -z "$CATEGORY" ] && CATEGORY="hand_landmark"
                          [ -z "$POSTPROCESSOR" ] && POSTPROCESSOR="hand_landmark" ;;
    *)                    print_error "Invalid task type: $TASK_TYPE"; usage ;;
esac

# ============================================================================
# Resolve reference: --base-model overrides --postprocessor
# ============================================================================
if [ -n "$BASE_MODEL" ]; then
    # --base-model: find the model directory and extract class names
    for lang in "${LANGS[@]}"; do
        case "$lang" in
            cpp) BASE_SRC_DIR="$CPP_SRC_DIR" ;;
            py)  BASE_SRC_DIR="$PY_SRC_DIR" ;;
        esac
        BASE_DIR=$(find "$BASE_SRC_DIR" -type d -name "$BASE_MODEL" | grep -v '/test' | head -1)
        if [ -n "$BASE_DIR" ]; then
            REF_DIR=$(echo "$BASE_DIR" | sed "s|${BASE_SRC_DIR}/||")
            REF_MODEL="$BASE_MODEL"

            if [ -f "$BASE_DIR/factory/${BASE_MODEL}_factory.hpp" ]; then
                CPP_FACTORY_CLASS=$(grep -oP 'class \K\w+Factory' "$BASE_DIR/factory/${BASE_MODEL}_factory.hpp" | head -1)
                CPP_MODEL_CLASS=$(grep -oP 'return "\K[^"]+' "$BASE_DIR/factory/${BASE_MODEL}_factory.hpp" | head -1)
            fi
            if [ -f "$BASE_DIR/factory/${BASE_MODEL}_factory.py" ]; then
                PY_FACTORY_CLASS=$(grep -oP 'class \K\w+Factory' "$BASE_DIR/factory/${BASE_MODEL}_factory.py" | head -1)
            fi
            print_info "Using base model: $BASE_MODEL → $REF_DIR"
            break
        fi
    done
    if [ -z "$REF_DIR" ]; then
        print_error "Base model not found: $BASE_MODEL"
        exit 1
    fi
else
    get_reference_info "$POSTPROCESSOR"
fi

# ============================================================================
# Generate new class names
# ============================================================================
CPP_NEW_MODEL_CLASS=$(to_pascal_case "$MODEL_NAME")
CPP_NEW_FACTORY_CLASS="${CPP_NEW_MODEL_CLASS}Factory"
PY_NEW_FACTORY_CLASS="$(to_py_class "$MODEL_NAME")Factory"

CPP_REF_HEADER_GUARD=$(echo "${REF_MODEL}_FACTORY_HPP" | tr '[:lower:]' '[:upper:]')
CPP_NEW_HEADER_GUARD=$(echo "${MODEL_NAME}_FACTORY_HPP" | tr '[:lower:]' '[:upper:]')
fi  # end: AUTO_ADD=false guard

# ============================================================================
# Generate C++ model package
# ============================================================================
generate_cpp() {
    local SRC_DIR="$CPP_SRC_DIR"
    local MODEL_DIR="${SRC_DIR}/${CATEGORY}/${MODEL_NAME}"

    # Validate reference
    local REF_FULL_DIR="${SRC_DIR}/${REF_DIR}"
    local REF_FACTORY_FILE="${REF_FULL_DIR}/factory/${REF_MODEL}_factory.hpp"
    local REF_SYNC_FILE="${REF_FULL_DIR}/${REF_MODEL}_sync.cpp"
    local REF_ASYNC_FILE="${REF_FULL_DIR}/${REF_MODEL}_async.cpp"

    # Glob fallback: factory/sync/async may use different file prefix
    if [ ! -f "$REF_FACTORY_FILE" ]; then
        local _found; _found=$(compgen -G "${REF_FULL_DIR}/factory/*_factory.hpp" 2>/dev/null | head -1)
        [ -n "$_found" ] && REF_FACTORY_FILE="$_found"
    fi
    if [ ! -f "$REF_SYNC_FILE" ]; then
        local _found; _found=$(compgen -G "${REF_FULL_DIR}/*_sync.cpp" 2>/dev/null | head -1)
        [ -n "$_found" ] && REF_SYNC_FILE="$_found"
    fi
    if [ ! -f "$REF_ASYNC_FILE" ]; then
        local _found; _found=$(compgen -G "${REF_FULL_DIR}/*_async.cpp" 2>/dev/null | head -1)
        [ -n "$_found" ] && REF_ASYNC_FILE="$_found"
    fi
    # Extract actual file prefix (may differ from REF_MODEL)
    local REF_ACTUAL="$REF_MODEL"
    if [ -f "$REF_FACTORY_FILE" ]; then
        local _base; _base=$(basename "$REF_FACTORY_FILE" _factory.hpp)
        [ -n "$_base" ] && REF_ACTUAL="$_base"
    fi

    if [ ! -f "$REF_FACTORY_FILE" ]; then
        print_error "C++ reference factory not found: ${REF_FULL_DIR}/factory/*_factory.hpp"
        return 1
    fi
    if [ ! -f "$REF_SYNC_FILE" ]; then
        print_error "C++ reference sync not found: ${REF_FULL_DIR}/*_sync.cpp"
        return 1
    fi

    if [ -d "$MODEL_DIR" ]; then
        print_warn "C++ directory already exists — skipping: $MODEL_DIR"
        return 0
    fi

    echo ""
    print_info "Creating C++ model package: ${MODEL_NAME}"
    print_info "  Task:          ${TASK_TYPE}"
    print_info "  Category:      ${CATEGORY}"
    print_info "  Postprocessor: ${POSTPROCESSOR}"
    print_info "  Reference:     ${REF_DIR}/${REF_MODEL}"
    print_info "  New class:     ${CPP_NEW_FACTORY_CLASS}"
    echo ""

    mkdir -p "$MODEL_DIR/factory"

    # --- factory header ---
    local FACTORY_FILE="${MODEL_DIR}/factory/${MODEL_NAME}_factory.hpp"
    sed \
        -e "s/${CPP_REF_HEADER_GUARD}/${CPP_NEW_HEADER_GUARD}/g" \
        -e "s/${CPP_FACTORY_CLASS}/${CPP_NEW_FACTORY_CLASS}/g" \
        -e "s/${REF_ACTUAL}_factory/${MODEL_NAME}_factory/g" \
        -e "s/\"${CPP_MODEL_CLASS}\"/\"${CPP_NEW_MODEL_CLASS}\"/g" \
        -e "s/@file ${REF_ACTUAL}_factory/@file ${MODEL_NAME}_factory/g" \
        -e "s/@brief ${CPP_MODEL_CLASS}/@brief ${CPP_NEW_MODEL_CLASS}/g" \
        "$REF_FACTORY_FILE" > "$FACTORY_FILE"
    print_file "$FACTORY_FILE (← ${REF_ACTUAL}_factory.hpp)"

    # --- sync.cpp ---
    local SYNC_FILE="${MODEL_DIR}/${MODEL_NAME}_sync.cpp"
    sed \
        -e "s/${REF_ACTUAL}_factory/${MODEL_NAME}_factory/g" \
        -e "s/${CPP_FACTORY_CLASS}/${CPP_NEW_FACTORY_CLASS}/g" \
        -e "s/${REF_ACTUAL}_sync/${MODEL_NAME}_sync/g" \
        -e "s/${CPP_MODEL_CLASS} synchronous/${CPP_NEW_MODEL_CLASS} synchronous/g" \
        -e "s/${CPP_MODEL_CLASS} Synchronous/${CPP_NEW_MODEL_CLASS} Synchronous/g" \
        "$REF_SYNC_FILE" > "$SYNC_FILE"
    print_file "$SYNC_FILE (← ${REF_ACTUAL}_sync.cpp)"

    # --- async.cpp ---
    if [ "$SYNC_ONLY" = false ] && [ -f "$REF_ASYNC_FILE" ]; then
        local ASYNC_FILE="${MODEL_DIR}/${MODEL_NAME}_async.cpp"
        sed \
            -e "s/${REF_ACTUAL}_factory/${MODEL_NAME}_factory/g" \
            -e "s/${CPP_FACTORY_CLASS}/${CPP_NEW_FACTORY_CLASS}/g" \
            -e "s/${REF_ACTUAL}_async/${MODEL_NAME}_async/g" \
            -e "s/${CPP_MODEL_CLASS} asynchronous/${CPP_NEW_MODEL_CLASS} asynchronous/g" \
            -e "s/${CPP_MODEL_CLASS} Asynchronous/${CPP_NEW_MODEL_CLASS} Asynchronous/g" \
            "$REF_ASYNC_FILE" > "$ASYNC_FILE"
        print_file "$ASYNC_FILE (← ${REF_ACTUAL}_async.cpp)"
    fi

    # --- CMakeLists.txt ---
    # Auto-discovery: directories with *_sync.cpp / *_async.cpp are detected automatically
    print_info "CMakeLists.txt uses auto-discovery — no manual update needed"

    # --- config.json ---
    local REF_CONFIG="${REF_FULL_DIR}/config.json"
    if [ -f "$REF_CONFIG" ]; then
        cp "$REF_CONFIG" "${MODEL_DIR}/config.json"
        print_file "${MODEL_DIR}/config.json (← reference config.json)"
    else
        # Generate minimal config.json template
        cat > "${MODEL_DIR}/config.json" << 'CJEOF'
{
}
CJEOF
        print_file "${MODEL_DIR}/config.json (empty template)"
    fi

    # Merge model-specific config from model_registry.json (overrides reference defaults)
    local REGISTRY_FILE="$DX_APP_ROOT/config/model_registry.json"
    if [ -f "$REGISTRY_FILE" ]; then
        python3 -c "
import json, sys
with open('$REGISTRY_FILE') as f:
    data = json.load(f)
models = data if isinstance(data, list) else data.get('models', [])
reg = next((m for m in models if m.get('model_name') == '${MODEL_NAME}'), None)
if reg and reg.get('config'):
    cfg_path = '${MODEL_DIR}/config.json'
    with open(cfg_path) as f:
        local_cfg = json.load(f)
    for k, v in reg['config'].items():
        local_cfg[k] = v
    with open(cfg_path, 'w') as f:
        json.dump(local_cfg, f, indent=4)
    print(f'  [INFO] Merged registry config: {list(reg[\"config\"].keys())}')
" 2>/dev/null || true
    fi

    echo ""
    print_success "C++ model package created: ${MODEL_NAME}"
    echo "Reference used: ${REF_DIR}/${REF_MODEL} (${CPP_FACTORY_CLASS})"
}

# ============================================================================
# Generate Python model package
# ============================================================================
generate_py() {
    local SRC_DIR="$PY_SRC_DIR"
    local MODEL_DIR="${SRC_DIR}/${CATEGORY}/${MODEL_NAME}"

    # Validate reference
    local REF_FULL_DIR="${SRC_DIR}/${REF_DIR}"
    local REF_FACTORY_PY="${REF_FULL_DIR}/factory/${REF_MODEL}_factory.py"
    local REF_INIT_PY="${REF_FULL_DIR}/factory/__init__.py"
    local REF_SYNC_PY="${REF_FULL_DIR}/${REF_MODEL}_sync.py"
    local REF_ASYNC_PY="${REF_FULL_DIR}/${REF_MODEL}_async.py"
    local REF_PY_FACTORY_CLASS=""
    local REF_PY_IMPORTED_FACTORY_CLASS=""

    # Fallback: if factory filename differs, find *_factory.py
    if [ ! -f "$REF_FACTORY_PY" ]; then
        local found_factory
        found_factory=$(find "${REF_FULL_DIR}/factory" -maxdepth 1 -name "*_factory.py" ! -name "__*" 2>/dev/null | head -1)
        if [ -n "$found_factory" ]; then
            REF_FACTORY_PY="$found_factory"
            print_warn "Using fallback factory: $(basename "$found_factory")"
        else
            print_error "Python reference factory not found: ${REF_FULL_DIR}/factory/*_factory.py"
            return 1
        fi
    fi

    REF_PY_FACTORY_CLASS=$(python3 - <<PY
import re
from pathlib import Path
path = Path(r'''$REF_FACTORY_PY''')
text = path.read_text(encoding='utf-8')
m = re.search(r'^class\s+(\w+)\s*\(', text, re.M)
print(m.group(1) if m else '')
PY
)
    REF_PY_IMPORTED_FACTORY_CLASS=$(python3 - <<PY
import re
from pathlib import Path
for candidate in [Path(r'''$REF_SYNC_PY'''), Path(r'''$REF_ASYNC_PY''')]:
    if not candidate.exists():
        continue
    text = candidate.read_text(encoding='utf-8')
    m = re.search(r'^from\s+factory\s+import\s+(\w+)', text, re.M)
    if m:
        print(m.group(1))
        break
else:
    print('')
PY
)
    if [ -z "$REF_PY_FACTORY_CLASS" ]; then
        REF_PY_FACTORY_CLASS="$PY_FACTORY_CLASS"
    fi
    if [ -z "$REF_PY_IMPORTED_FACTORY_CLASS" ]; then
        REF_PY_IMPORTED_FACTORY_CLASS="$REF_PY_FACTORY_CLASS"
    fi

    if [ -d "$MODEL_DIR" ]; then
        print_warn "Python directory already exists — skipping: $MODEL_DIR"
        return 0
    fi

    echo ""
    print_info "Creating Python model package: ${MODEL_NAME}"
    print_info "  Task:          ${TASK_TYPE}"
    print_info "  Category:      ${CATEGORY}"
    print_info "  Postprocessor: ${POSTPROCESSOR}"
    print_info "  Reference:     ${REF_DIR}/${REF_MODEL}"
    print_info "  New class:     ${PY_NEW_FACTORY_CLASS}"
    echo ""

    mkdir -p "$MODEL_DIR/factory"

    # --- factory.py ---
    local PY_SAFE_NAME
    PY_SAFE_NAME=$(to_py_safe_module "$MODEL_NAME")
    local FACTORY_FILE="${MODEL_DIR}/factory/${PY_SAFE_NAME}_factory.py"
    sed \
        -e "s/${REF_PY_FACTORY_CLASS}/${PY_NEW_FACTORY_CLASS}/g" \
        -e "s/${REF_MODEL}_factory/${MODEL_NAME}_factory/g" \
        -e "s/\"${REF_MODEL}\"/\"${MODEL_NAME}\"/g" \
        -e "s/${REF_MODEL^} Factory/${MODEL_NAME^} Factory/g" \
        -e "s/creating ${REF_MODEL^}/creating ${MODEL_NAME^}/g" \
        "$REF_FACTORY_PY" > "$FACTORY_FILE"
    print_file "$FACTORY_FILE (← ${REF_MODEL}_factory.py)"

    # --- factory/__init__.py ---
    # Always generate fresh __init__.py to avoid filename mismatch issues
    local INIT_FILE="${MODEL_DIR}/factory/__init__.py"
    echo "from .${PY_SAFE_NAME}_factory import ${PY_NEW_FACTORY_CLASS}" > "$INIT_FILE"
    print_file "$INIT_FILE"

    # --- sync.py ---
    local SYNC_FILE="${MODEL_DIR}/${MODEL_NAME}_sync.py"
    sed \
        -e "s/${REF_PY_IMPORTED_FACTORY_CLASS}/${PY_NEW_FACTORY_CLASS}/g" \
        -e "s/${REF_PY_FACTORY_CLASS}/${PY_NEW_FACTORY_CLASS}/g" \
        -e "s/${REF_MODEL}_sync/${MODEL_NAME}_sync/g" \
        -e "s/${REF_MODEL^} Sync/${MODEL_NAME^} Sync/g" \
        -e "s/${REF_MODEL^} Synchronous/${MODEL_NAME^} Synchronous/g" \
        "$REF_SYNC_PY" > "$SYNC_FILE"
    chmod +x "$SYNC_FILE"
    print_file "$SYNC_FILE (← ${REF_MODEL}_sync.py)"

    # --- async.py ---
    if [ "$SYNC_ONLY" = false ] && [ -f "$REF_ASYNC_PY" ]; then
        local ASYNC_FILE="${MODEL_DIR}/${MODEL_NAME}_async.py"
        sed \
            -e "s/${REF_PY_IMPORTED_FACTORY_CLASS}/${PY_NEW_FACTORY_CLASS}/g" \
            -e "s/${REF_PY_FACTORY_CLASS}/${PY_NEW_FACTORY_CLASS}/g" \
            -e "s/${REF_MODEL}_async/${MODEL_NAME}_async/g" \
            -e "s/${REF_MODEL^} Async/${MODEL_NAME^} Async/g" \
            -e "s/${REF_MODEL^} Asynchronous/${MODEL_NAME^} Asynchronous/g" \
            "$REF_ASYNC_PY" > "$ASYNC_FILE"
        chmod +x "$ASYNC_FILE"
        print_file "$ASYNC_FILE (← ${REF_MODEL}_async.py)"
    fi

    # --- cpp_postprocess variants (copy from reference, adapting factory class) ---
    # Auto-generated models inherit the reference model's cpp_postprocess.py directly:
    #   - References using a real C++ class (e.g. YOLOv8PostProcess) → derived models
    #     get the same C++ class automatically, enabling native dx_postprocess.so binding.
    #   - References that use PythonFallbackPostProcess (e.g. dncnn, retinaface) → derived
    #     models also inherit the fallback, which is correct for models that require the
    #     input image context (DnCNN).
    # Only the factory class name and filename references are substituted via sed.
    for variant_type in sync async; do
        local REF_VARIANT="${REF_FULL_DIR}/${REF_MODEL}_${variant_type}_cpp_postprocess.py"
        [ -f "$REF_VARIANT" ] || continue

        # Skip async if SYNC_ONLY
        [ "$variant_type" = "async" ] && [ "$SYNC_ONLY" = true ] && continue

        local CPP_PP_FILE="${MODEL_DIR}/${MODEL_NAME}_${variant_type}_cpp_postprocess.py"

        # Copy reference and adapt: factory class name + filename references
        sed \
            -e "s/${REF_PY_IMPORTED_FACTORY_CLASS}/${PY_NEW_FACTORY_CLASS}/g" \
            -e "s/${REF_PY_FACTORY_CLASS}/${PY_NEW_FACTORY_CLASS}/g" \
            -e "s/${REF_MODEL}_${variant_type}_cpp_postprocess/${MODEL_NAME}_${variant_type}_cpp_postprocess/g" \
            -e "s/${REF_MODEL^} Sync/${MODEL_NAME^} Sync/g" \
            -e "s/${REF_MODEL^} Async/${MODEL_NAME^} Async/g" \
            -e "s/${REF_MODEL^} Synchronous/${MODEL_NAME^} Synchronous/g" \
            -e "s/${REF_MODEL^} Asynchronous/${MODEL_NAME^} Asynchronous/g" \
            "$REF_VARIANT" > "$CPP_PP_FILE"
        chmod +x "$CPP_PP_FILE"

        # Report which C++ class was inherited (or fallback)
        if grep -q "PythonFallbackPostProcess" "$CPP_PP_FILE"; then
            print_file "${MODEL_NAME}_${variant_type}_cpp_postprocess.py (← Python fallback)"
        else
            local _CPP_CLASS
            _CPP_CLASS=$(grep -oP 'from dx_postprocess import \K\w+' "$CPP_PP_FILE" | head -1)
            print_file "${MODEL_NAME}_${variant_type}_cpp_postprocess.py (← ${_CPP_CLASS})"
        fi
    done

    # --- config.json ---
    local REF_CONFIG="${REF_FULL_DIR}/config.json"
    if [ -f "$REF_CONFIG" ]; then
        cp "$REF_CONFIG" "${MODEL_DIR}/config.json"
        print_file "${MODEL_DIR}/config.json (← reference config.json)"
    else
        # Generate minimal config.json template
        cat > "${MODEL_DIR}/config.json" << 'CJEOF'
{
}
CJEOF
        print_file "${MODEL_DIR}/config.json (empty template)"
    fi

    # Merge model-specific config from model_registry.json (overrides reference defaults)
    local REGISTRY_FILE="$DX_APP_ROOT/config/model_registry.json"
    if [ -f "$REGISTRY_FILE" ]; then
        python3 -c "
import json, sys
with open('$REGISTRY_FILE') as f:
    data = json.load(f)
models = data if isinstance(data, list) else data.get('models', [])
reg = next((m for m in models if m.get('model_name') == '${MODEL_NAME}'), None)
if reg and reg.get('config'):
    cfg_path = '${MODEL_DIR}/config.json'
    with open(cfg_path) as f:
        local_cfg = json.load(f)
    for k, v in reg['config'].items():
        local_cfg[k] = v
    with open(cfg_path, 'w') as f:
        json.dump(local_cfg, f, indent=4)
    print(f'  [INFO] Merged registry config: {list(reg[\"config\"].keys())}')
" 2>/dev/null || true
    fi

    echo ""
    print_success "Python model package created: ${MODEL_NAME}"
    echo "Reference used: ${REF_DIR}/${REF_MODEL} (${PY_FACTORY_CLASS})"
}

# ============================================================================
# --auto-add: Scan assets/models/ and auto-register new .dxnn files in GUI
# ============================================================================
if [ "$AUTO_ADD" = true ]; then
    CONF_FILE="$DX_APP_ROOT/config/test_models.conf"
    MODELS_DIR="$DX_APP_ROOT/assets/models"

    print_info "Auto-add: scanning ${MODELS_DIR} for unregistered .dxnn files"

    export DX_APP_ROOT AUTO_ADD_MANIFEST
    NEW_MODELS=$(DX_APP_ROOT="$DX_APP_ROOT" AUTO_ADD_MANIFEST="$AUTO_ADD_MANIFEST" python3 - << 'PYEOF'
import sys, json, os
from pathlib import Path

DX_APP   = Path(os.environ["DX_APP_ROOT"])
MODELS   = DX_APP / "assets" / "models"
CONF     = DX_APP / "config" / "test_models.conf"
MANIFEST = os.environ.get("AUTO_ADD_MANIFEST", "")
CPP_SRC  = DX_APP / "src" / "cpp_example"
PY_SRC   = DX_APP / "src" / "python_example"

# Parse conf: fname -> (model_name, category)  and  model_name -> (category, fpath)
conf_by_file = {}   # dxnn filename -> (model_name, category, fpath)
conf_by_name = {}   # model_name   -> (category, fpath)
for line in CONF.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"): continue
    parts = line.split("\t")
    if len(parts) >= 3:
        mn, cat, fp = parts[0].strip(), parts[1].strip(), parts[2].strip()
        conf_by_file[Path(fp).name] = (mn, cat, fp)
        conf_by_name[mn] = (cat, fp)

# manifest: dxnn filename -> {category, class_name}
manifest_data = {}
if MANIFEST and Path(MANIFEST).exists():
    from urllib.parse import urlparse
    for m in json.loads(Path(MANIFEST).read_text()):
        fname = Path(urlparse(m["dxnn_url"]).path).name
        manifest_data[fname] = {
            "category":   m.get("category", ""),
            "class_name": m.get("name", ""),
        }

def has_src(mn):
    for root in [CPP_SRC, PY_SRC]:
        for cat_dir in root.iterdir():
            if cat_dir.is_dir() and (cat_dir / mn).is_dir():
                return True
    return False

# Map (category, model_name) -> postprocessor for known families
def infer_pp(mn, category):
    n = mn.lower()
    cat = category.lower()
    # Pose
    if "pose" in n: return "yolov8pose" if "8" in n else "yolov5pose"
    # Segmentation
    if "seg" in n or cat == "instance_segmentation":
        return "yolov5seg" if "5" in n else "yolov8seg"
    if cat == "semantic_segmentation":
        return "bisenetv1" if "bisenet" in n else "deeplabv3"
    # Detection families
    if cat == "object_detection":
        if "centernet" in n:    return "centernet"
        if "damoyolo" in n:     return "damoyolo"
        if "nanodet" in n:      return "nanodet"
        if "ssd" in n:          return "ssd"
        if "yolov26" in n or "yolo26" in n: return "yolov26"
        if "yolox" in n:        return "yolox"
        if "yolov11" in n:      return "yolov11"
        if "yolov10" in n:      return "yolov10"
        if "yolov9" in n:       return "yolov9"
        if "yolov8" in n:       return "yolov8"
        if "yolov7" in n:       return "yolov7"
        if "yolov6" in n:       return "yolov8"   # same postprocessor
        if "yolov5" in n:       return "yolov5"
        if "yolov3" in n:       return "yolov5"   # same postprocessor
        return "yolov8"
    # Face detection
    if cat == "face_detection":
        if "scrfd" in n: return "scrfd"
        if "yolov7" in n or "yolov5" in n: return "yolov5face"
        return "scrfd"
    # Depth
    if cat == "depth_estimation":   return "fastdepth"
    # Image denoising
    if cat == "image_denoising":    return "dncnn"
    # Super resolution
    if cat == "super_resolution":   return "espcn"
    # Embedding
    if cat == "embedding":
        if "text" in n: return "clip_text"
        if "osnet" in n: return "arcface"
        return "clip_image"
    # PPU
    if cat == "ppu":
        if "yolov7" in n: return "yolov7_ppu"
        if "pose" in n:   return "yolov5pose_ppu"
        return "yolov5_ppu"
    # Hand landmark
    if cat == "hand_landmark":      return "hand_landmark"
    # Classification (default)
    return "efficientnet"

# Map category -> task_type for add_model.sh
CAT_TO_TASK = {
    "classification":           "classification",
    "object_detection":         "detection",
    "face_detection":           "face_detection",
    "pose_estimation":          "pose",
    "instance_segmentation":    "instance_segmentation",
    "semantic_segmentation":    "semantic_segmentation",
    "depth_estimation":         "depth_estimation",
    "image_denoising":          "image_denoising",
    "super_resolution":         "super_resolution",
    "image_enhancement":        "image_enhancement",
    "embedding":                "embedding",
    "ppu":                      "ppu",
    "hand_landmark":            "hand_landmark",
    "obb_detection":            "obb",
}

actual_dxnn = {f.name for f in MODELS.glob("*.dxnn")}

import re as _re
from collections import Counter

def _normalize(s):
    """Convert a name string to a valid directory key (lowercase, underscores)."""
    return _re.sub(r'_+', '_', s.lower().replace("-","_").replace(".","_").replace(" ","_")).strip("_")

def _bare(s):
    """Strip trailing numeric version suffix from a normalized key (e.g. 'dncnn_2' -> 'dncnn', 'yolo26n_1' -> 'yolo26n')."""
    return _re.sub(r'_\d+$', '', s)

# Pre-scan: derive keys for all new dxnn files
_new_dxnns = [d for d in sorted(MODELS.glob("*.dxnn")) if d.name not in conf_by_file]

def _full_class_key(dxnn):
    """Full normalized class name key (e.g. 'dncnn_2', 'yolo26n_1')."""
    entry = manifest_data.get(dxnn.name, {})
    cn = entry.get("class_name", "") if isinstance(entry, dict) else ""
    return _normalize(cn) if cn else _normalize(dxnn.stem)

# Count how many files share the same BARE key (class name with version suffix stripped)
_bare_counts = Counter(_bare(_full_class_key(d)) for d in _new_dxnns)

# 1) New dxnn files not in conf at all
for dxnn in _new_dxnns:
    full_key  = _full_class_key(dxnn)
    bare_key  = _bare(full_key)
    # Multiple files share same bare class name → use full key (includes _2/_3 to distinguish)
    # Single file for this class name → strip version suffix (e.g. yolo26n_1 → yolo26n)
    key = full_key if _bare_counts[bare_key] > 1 else bare_key
    entry = manifest_data.get(dxnn.name, {})
    cat_hint = entry.get("category", "") if isinstance(entry, dict) else entry
    ch = cat_hint.lower()
    if "pose" in ch:                  category = "pose_estimation"
    elif "instance" in ch and "segmentation" in ch:
                                      category = "instance_segmentation"
    elif "segmentation" in ch:        category = "semantic_segmentation"
    elif "face" in ch and "detection" in ch:
                                      category = "face_detection"
    elif "detection" in ch:           category = "object_detection"
    elif "classification" in ch:      category = "classification"
    elif "depth" in ch:               category = "depth_estimation"
    elif "denoi" in ch:               category = "image_denoising"
    else:
        # filename-based fallback
        f = dxnn.name.lower()
        if "pose" in f:               category = "pose_estimation"
        elif "seg" in f:              category = "instance_segmentation"
        elif "depth" in f or "fastdepth" in f:
                                      category = "depth_estimation"
        elif "dncnn" in f:            category = "image_denoising"
        elif any(x in f for x in ["yolo","ssd","nanodet","retinaface"]): category = "object_detection"
        else:                         category = "classification"
    task = CAT_TO_TASK.get(category, "classification")
    pp   = infer_pp(key, category)
    print(f"{dxnn.name}\t{key}\t{task}\t{pp}\t{category}\t1")

# 2) Conf-registered models with dxnn present but no source directory
for mn, (cat, fpath) in sorted(conf_by_name.items()):
    fname = Path(fpath).name
    if fname not in actual_dxnn: continue   # dxnn not in models/ → skip
    if has_src(mn): continue                # already has source → skip
    task = CAT_TO_TASK.get(cat, "classification")
    pp   = infer_pp(mn, cat)
    print(f"{fname}\t{mn}\t{task}\t{pp}\t{cat}\t0")
PYEOF
    )

    if [ -z "$NEW_MODELS" ]; then
        print_info "No models to process — all .dxnn files already have source packages."
        exit 0
    fi

    total=$(echo "$NEW_MODELS" | grep -c .)
    print_info "Models to process: ${total}"

    added=0; src_only=0
    while IFS=$'\t' read -r fname model_key task_type postprocessor category need_conf; do
        [ -z "$fname" ] && continue

        echo ""
        echo "──────────────────────────────────────────────────"
        print_info "Processing: ${model_key} (${category}) ← ${fname}"

        # Set globals required by generate_cpp / generate_py
        MODEL_NAME="$model_key"
        TASK_TYPE="$task_type"
        CATEGORY="$category"
        POSTPROCESSOR="$postprocessor"
        SYNC_ONLY=false
        BASE_MODEL=""
        LANGS=("cpp" "py")
        CPP_MODEL_DIR="${CPP_SRC_DIR}/${CATEGORY}/${MODEL_NAME}"
        PY_MODEL_DIR="${PY_SRC_DIR}/${CATEGORY}/${MODEL_NAME}"

        # Resolve reference info (sets REF_DIR, REF_MODEL, *_CLASS globals)
        get_reference_info "$POSTPROCESSOR"

        # Compute new class name globals
        CPP_NEW_MODEL_CLASS=$(to_pascal_case "$MODEL_NAME")
        CPP_NEW_FACTORY_CLASS="${CPP_NEW_MODEL_CLASS}Factory"
        PY_NEW_FACTORY_CLASS="$(to_py_class "$MODEL_NAME")Factory"
        CPP_REF_HEADER_GUARD=$(echo "${REF_MODEL}_FACTORY_HPP" | tr '[:lower:]' '[:upper:]')
        CPP_NEW_HEADER_GUARD=$(echo "${MODEL_NAME}_FACTORY_HPP" | tr '[:lower:]' '[:upper:]')

        # Generate source code directories (skip if already exist)
        generate_cpp
        generate_py

        # Register in test_models.conf only if not already there
        if [ "$need_conf" = "1" ]; then
            printf "%s\t%s\tassets/models/%s\n" "$model_key" "$category" "$fname" >> "$CONF_FILE"
            print_success "Registered in conf: ${model_key} → ${category} → assets/models/${fname}"
            added=$((added + 1))
        else
            print_success "Source created: ${model_key} (already in conf)"
            src_only=$((src_only + 1))
        fi

    done <<< "$NEW_MODELS"

    echo ""
    echo "══════════════════════════════════════════════════"
    print_success "Auto-add complete: ${added} new conf entry(s), ${src_only} source-only"
    echo "══════════════════════════════════════════════════"
    exit 0
fi

# ============================================================================
# Generate packages
# ============================================================================
echo ""
echo "=============================================="
echo "  Creating model: ${MODEL_NAME} (lang: ${LANG_MODE})"
echo "=============================================="

CPP_MODEL_DIR="${CPP_SRC_DIR}/${CATEGORY}/${MODEL_NAME}"
PY_MODEL_DIR="${PY_SRC_DIR}/${CATEGORY}/${MODEL_NAME}"

for lang in "${LANGS[@]}"; do
    case "$lang" in
        cpp) generate_cpp ;;
        py)  generate_py ;;
    esac
done

# ============================================================================
# --verify: Build + Run inference
# ============================================================================
if [ "$VERIFY" = true ]; then
    set +e
    echo ""
    echo "========================================"
    echo "  AUTO-VERIFY: ${MODEL_NAME} (lang: ${LANG_MODE})"
    echo "  Started: $(date)"
    echo "========================================"

    PASS=0; FAIL=0; SKIP=0

    run_test() {
        local name="$1" cmd="$2"
        echo -ne "${BLUE}[RUN]${NC} ${name}... "
        local output
        output=$(eval "$cmd" 2>&1)
        local rc=$?
        if [ $rc -eq 0 ]; then
            echo -e "${GREEN}PASS${NC}"
            local perf
            perf=$(echo "$output" | grep -A 100 'PERFORMANCE SUMMARY\|PROCESSING SUMMARY' | head -20)
            if [ -n "$perf" ]; then
                echo "$perf" | while IFS= read -r line; do
                    echo "        $line"
                done
            fi
            PASS=$((PASS + 1))
        else
            echo -e "${RED}FAIL${NC} (exit=$rc)"
            echo "$output" | tail -5
            FAIL=$((FAIL + 1))
        fi
    }

    # --- Resolve paths ---
    [ -z "$VERIFY_IMAGE" ] && VERIFY_IMAGE="${DX_APP_ROOT}/sample/img/sample_kitchen.jpg"
    [ "$NO_VIDEO" = false ] && [ -z "$VERIFY_VIDEO" ] && VERIFY_VIDEO=$(find -L "${DX_APP_ROOT}/assets/videos" -type f -name "*.mov" 2>/dev/null | head -1)

    if [ -z "$VERIFY_MODEL" ]; then
        print_error "No model specified. Use --model <path>"
        exit 1
    fi

    # Convert to absolute paths
    if [[ "$VERIFY_MODEL" != /* ]]; then
        VERIFY_MODEL=$(realpath "$VERIFY_MODEL" 2>/dev/null || realpath "${DX_APP_ROOT}/${VERIFY_MODEL}" 2>/dev/null || echo "${DX_APP_ROOT}/${VERIFY_MODEL}")
    fi
    if [[ "$VERIFY_IMAGE" != /* ]]; then
        VERIFY_IMAGE=$(realpath "$VERIFY_IMAGE" 2>/dev/null || realpath "${DX_APP_ROOT}/${VERIFY_IMAGE}" 2>/dev/null || echo "${DX_APP_ROOT}/${VERIFY_IMAGE}")
    fi
    if [[ -n "$VERIFY_VIDEO" && "$VERIFY_VIDEO" != /* ]]; then
        VERIFY_VIDEO=$(realpath "$VERIFY_VIDEO" 2>/dev/null || realpath "${DX_APP_ROOT}/${VERIFY_VIDEO}" 2>/dev/null || echo "${DX_APP_ROOT}/${VERIFY_VIDEO}")
    fi

    # ================================================================
    # C++ Verify: Build + Run
    # ================================================================
    for lang in "${LANGS[@]}"; do
        if [ "$lang" = "cpp" ]; then
            echo ""
            echo -e "${BLUE}━━━ C++ Build ━━━${NC}"
            BUILD_DIR="${DX_APP_ROOT}/build"
            print_info "cmake reconfigure..."
            mkdir -p "$BUILD_DIR"
            cmake -S "$DX_APP_ROOT" -B "$BUILD_DIR" > /dev/null 2>&1

            SYNC_TARGET="${MODEL_NAME}_sync"
            ASYNC_TARGET="${MODEL_NAME}_async"

            run_test "cpp_build_sync" "cd '$BUILD_DIR' && make $SYNC_TARGET -j\$(nproc) 2>&1"

            if [ "$SYNC_ONLY" = false ]; then
                REF_ASYNC_FILE="${CPP_SRC_DIR}/${REF_DIR}/${REF_MODEL}_async.cpp"
                if [ -f "$REF_ASYNC_FILE" ]; then
                    run_test "cpp_build_async" "cd '$BUILD_DIR' && make $ASYNC_TARGET -j\$(nproc) 2>&1"
                fi
            fi

            SYNC_BIN=$(find "$BUILD_DIR" -name "$SYNC_TARGET" -type f -executable 2>/dev/null | head -1)
            ASYNC_BIN=$(find "$BUILD_DIR" -name "$ASYNC_TARGET" -type f -executable 2>/dev/null | head -1)

            DISPLAY_PREFIX=""
            if ! xdpyinfo >/dev/null 2>&1; then
                if command -v xvfb-run >/dev/null 2>&1; then
                    DISPLAY_PREFIX="xvfb-run --auto-servernum"
                else
                    print_warn "No display and no xvfb-run. GUI tests may fail."
                fi
            fi

            echo ""
            echo -e "${BLUE}━━━ C++ Inference ━━━${NC}"
            cd "$DX_APP_ROOT"

            if [ -n "$SYNC_BIN" ]; then
                run_test "cpp_sync_image" \
                    "cd '$DX_APP_ROOT' && $DISPLAY_PREFIX '$SYNC_BIN' -m '$VERIFY_MODEL' -i '$VERIFY_IMAGE' 2>&1"
                if [ -n "$VERIFY_VIDEO" ]; then
                    run_test "cpp_sync_video" \
                        "cd '$DX_APP_ROOT' && $DISPLAY_PREFIX '$SYNC_BIN' -m '$VERIFY_MODEL' -v '$VERIFY_VIDEO' 2>&1"
                fi
            else
                echo -e "${YELLOW}[SKIP]${NC} C++ sync binary not found"
                SKIP=$((SKIP + 1))
            fi

            if [ -n "$ASYNC_BIN" ] && [ -n "$VERIFY_VIDEO" ]; then
                run_test "cpp_async_video" \
                    "cd '$DX_APP_ROOT' && $DISPLAY_PREFIX '$ASYNC_BIN' -m '$VERIFY_MODEL' -v '$VERIFY_VIDEO' 2>&1"
            fi
        fi

        # ================================================================
        # Python Verify: Run
        # ================================================================
        if [ "$lang" = "py" ]; then
            PYTHON_CMD="python3"
            # Prefer DX_PYTHON env var, then search common venv locations
            if [ -n "${DX_PYTHON:-}" ] && [ -f "${DX_PYTHON}" ]; then
                PYTHON_CMD="$DX_PYTHON"
            else
                for _venv in \
                    "${VIRTUAL_ENV:-__none__}/bin/python3" \
                    "${DX_APP_ROOT}/../venv-dx-runtime/bin/python3" \
                    "${DX_APP_ROOT}/../../venv-dx-runtime/bin/python3" \
                    "${DX_APP_ROOT}/../../dx-venv/bin/python3" \
                    "${DX_APP_ROOT}/../../../dx-venv/bin/python3" \
                    "${HOME}/dx-venv/bin/python3"; do
                    if [ -f "$_venv" ]; then
                        PYTHON_CMD="$_venv"
                        break
                    fi
                done
            fi

            resolve_py_entry() {
                local mode_suffix="$1"
                local exact_path="${PY_MODEL_DIR}/${MODEL_NAME}_${mode_suffix}.py"

                if [ -f "$exact_path" ]; then
                    echo "$exact_path"
                    return 0
                fi

                local fallback_path
                fallback_path=$(find "$PY_MODEL_DIR" -maxdepth 1 -type f -name "*_${mode_suffix}.py" | sort | head -1)
                if [ -n "$fallback_path" ]; then
                    echo -e "${YELLOW}[WARN]${NC} Using fallback Python entry for ${MODEL_NAME}: $(basename "$fallback_path")" >&2
                    echo "$fallback_path"
                    return 0
                fi

                return 1
            }

            PY_SYNC_SCRIPT=$(resolve_py_entry "sync")
            PY_ASYNC_SCRIPT=$(resolve_py_entry "async" 2>/dev/null || true)
            PY_SYNC_CPP_PP_SCRIPT=$(resolve_py_entry "sync_cpp_postprocess" 2>/dev/null || true)

            # Display detection: if no display is available, force --no-display
            PY_DISPLAY_ARGS=""
            if ! xdpyinfo >/dev/null 2>&1; then
                PY_DISPLAY_ARGS="--no-display"
            fi

            echo ""
            echo -e "${BLUE}━━━ Python Inference ━━━${NC}"

            if [ -z "$PY_SYNC_SCRIPT" ]; then
                echo -e "${RED}[ERROR]${NC} Python sync entry not found in ${PY_MODEL_DIR}"
                FAIL=$((FAIL + 1))
            else
                run_test "py_sync_image" \
                    "cd '$DX_APP_ROOT' && $PYTHON_CMD '$PY_SYNC_SCRIPT' -m '$VERIFY_MODEL' -i '$VERIFY_IMAGE' $PY_DISPLAY_ARGS 2>&1"
            fi

            if [ -n "$VERIFY_VIDEO" ] && [ -n "$PY_SYNC_SCRIPT" ]; then
                run_test "py_sync_video" \
                    "cd '$DX_APP_ROOT' && $PYTHON_CMD '$PY_SYNC_SCRIPT' -m '$VERIFY_MODEL' -v '$VERIFY_VIDEO' $PY_DISPLAY_ARGS 2>&1"
            fi

            if [ "$SYNC_ONLY" = false ] && [ -n "$PY_ASYNC_SCRIPT" ] && [ -n "$VERIFY_VIDEO" ]; then
                run_test "py_async_video" \
                    "cd '$DX_APP_ROOT' && $PYTHON_CMD '$PY_ASYNC_SCRIPT' -m '$VERIFY_MODEL' -v '$VERIFY_VIDEO' $PY_DISPLAY_ARGS 2>&1"
            fi

            if [ -n "$PY_SYNC_CPP_PP_SCRIPT" ]; then
                run_test "py_sync_cpp_pp_image" \
                    "cd '$DX_APP_ROOT' && $PYTHON_CMD '$PY_SYNC_CPP_PP_SCRIPT' -m '$VERIFY_MODEL' -i '$VERIFY_IMAGE' $PY_DISPLAY_ARGS 2>&1"
            fi
        fi
    done

    echo ""
    echo "========================================"
    echo -e "  ${GREEN}PASS: ${PASS}${NC}  ${RED}FAIL: ${FAIL}${NC}  ${YELLOW}SKIP: ${SKIP}${NC}"
    echo "  Finished: $(date)"
    echo "========================================"

    if [ $FAIL -gt 0 ]; then
        [ "$GIT_PUSH" = true ] && print_warn "Verification failed — skipping git push."
        exit 1
    fi

    # ============================================================================
    # --git-push: Auto commit & push after successful verification
    # ============================================================================
    if [ "$GIT_PUSH" = true ]; then
        echo ""
        echo -e "${BLUE}━━━ Git Push ━━━${NC}"

        GIT_ROOT=$(git -C "$DX_APP_ROOT" rev-parse --show-toplevel 2>/dev/null)
        if [ -z "$GIT_ROOT" ]; then
            print_error "Not a git repository"
            exit 1
        fi
        cd "$GIT_ROOT"

        GIT_REMOTE="origin"
        if git remote | grep -q '^personal$'; then
            GIT_REMOTE="personal"
        fi
        GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

        if [ "$GIT_BRANCH" = "HEAD" ]; then
            print_error "Detached HEAD state. Please checkout a branch first."
            exit 1
        fi

        COMMIT_MSG="feat: add model ${MODEL_NAME} (${TASK_TYPE}, ${LANG_MODE}) - verified"

        echo ""
        print_info "Remote : ${GIT_REMOTE}"
        print_info "Branch : ${GIT_BRANCH}"
        print_info "Commit : ${COMMIT_MSG}"
        echo ""

        # Stage files
        for lang in "${LANGS[@]}"; do
            case "$lang" in
                cpp)
                    git add "$CPP_MODEL_DIR"
                    ;;
                py)
                    git add "$PY_MODEL_DIR"
                    ;;
            esac
        done

        echo -e "${CYAN}Files to be committed:${NC}"
        git diff --cached --name-only | while IFS= read -r f; do
            echo -e "  ${CYAN}→ $f${NC}"
        done
        echo ""

        if [ "$AUTO_YES" = true ]; then
            REPLY="y"
        else
            echo -ne "${YELLOW}Push to ${GIT_REMOTE}/${GIT_BRANCH}? [y/N]:${NC} "
            read -r REPLY
        fi

        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            git commit -m "$COMMIT_MSG"
            if git push "$GIT_REMOTE" "$GIT_BRANCH" 2>&1; then
                print_success "Pushed to ${GIT_REMOTE}/${GIT_BRANCH}"
            else
                print_error "Push failed"
                exit 1
            fi
        else
            print_warn "Git push cancelled (changes are staged but not committed)"
            git reset HEAD > /dev/null 2>&1
        fi
    fi

    exit 0
fi
