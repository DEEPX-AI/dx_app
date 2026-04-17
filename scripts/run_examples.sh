#!/bin/bash
# =============================================================================
# run_examples.sh - DX-APP Unified Example Runner
# =============================================================================
# Automatically discovers and runs C++/Python examples, then reports results.
# Reads model file paths and category info from config/test_models.conf.
#
# Usage:
#   scripts/run_examples.sh                              # Interactive mode
#   scripts/run_examples.sh --lang cpp                   # CLI: C++ only
#   scripts/run_examples.sh --lang py                    # CLI: Python only
#   scripts/run_examples.sh --lang both                  # CLI: Explicit all
#   scripts/run_examples.sh --lang cpp --display         # CLI: GUI display mode
#   scripts/run_examples.sh --lang cpp --no-video        # CLI: Image tests only
#   scripts/run_examples.sh --lang py  --video-only      # CLI: Video tests only
#   scripts/run_examples.sh --lang cpp --sync-only       # CLI: Sync tests only (C++)
#   scripts/run_examples.sh --lang cpp --async-only      # CLI: Async tests only (C++)
#   scripts/run_examples.sh --filter yolov5              # CLI: Name filter
# =============================================================================

set +e  # continue on error

# ============================================================
# Resolve paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Auto-activate venv-dx-runtime if python is not available
if ! command -v python &>/dev/null; then
    VENV_CANDIDATES=(
        "${PROJECT_ROOT}/../dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate"
    )
    for venv_activate in "${VENV_CANDIDATES[@]}"; do
        if [[ -f "$venv_activate" ]]; then
            # shellcheck source=/dev/null
            source "$venv_activate"
            echo "Auto-activated venv: $venv_activate" >&2
            break
        fi
    done
fi

# BUILD_DIR can be provided from the environment (e.g. BUILD_DIR=bin ./scripts/run_examples.sh)
# If not provided, try a list of common build output locations and pick the first existing one.
if [ -z "${BUILD_DIR:-}" ]; then
    CANDIDATE_DIRS=(
        "bin"
        "build_x86_64/release/bin"
        "build_aarch64/release/bin"
        "build_x86_64/bin"
        "build_aarch64/bin"
        "src/cpp_example/build"
        "build/release/bin"
        "build/bin"
    )
    BUILD_DIR=""
    for d in "${CANDIDATE_DIRS[@]}"; do
        if [ -d "${PROJECT_ROOT}/${d}" ]; then
            BUILD_DIR="${d}"
            break
        fi
    done
    # Fallback to the original path if nothing found
    if [ -z "${BUILD_DIR}" ]; then
        BUILD_DIR="src/cpp_example/build"
    fi
fi
echo "Using C++ build directory: ${BUILD_DIR}" >&2
PY_BASE="src/python_example"
CONFIG_FILE="${PROJECT_ROOT}/config/test_models.conf"

# Source common utilities
if [[ -f "$SCRIPT_DIR/color_env.sh" ]]; then
    source "$SCRIPT_DIR/color_env.sh"
else
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
fi

# ============================================================
# Category → Default Input Mapping
# ============================================================
declare -A CATEGORY_IMAGE CATEGORY_VIDEO CATEGORY_DISPLAY
CATEGORY_IMAGE=(
    [object_detection]="sample/img/sample_street.jpg"
    [face_detection]="sample/img/sample_face.jpg"
    [pose_estimation]="sample/img/sample_people.jpg"
    [obb_detection]="sample/dota8_test/P0284.png"
    [classification]="sample/img/sample_dog.jpg"
    [instance_segmentation]="sample/img/sample_street.jpg"
    [semantic_segmentation]="sample/img/sample_parking.jpg"
    [depth_estimation]="sample/img/sample_horse.jpg"
    [image_denoising]="sample/img/sample_denoising.jpg"
    [super_resolution]="sample/img/sample_superresolution.png"
    [image_enhancement]="sample/img/sample_lowlight.jpg"
    [embedding]="sample/img/face_pair"
    [attribute_recognition]="sample/img/sample_person_a1.jpg"
    [reid]="sample/img/person_pair"
    [ppu]="sample/img/sample_street.jpg"
    [hand_landmark]="sample/img/sample_hand.jpg"
    [face_alignment]="sample/img/sample_face_a1.jpg"
)
CATEGORY_VIDEO=(
    [object_detection]="assets/videos/blackbox-city-road.mp4"
    [face_detection]="assets/videos/dance-solo.mov"
    [pose_estimation]="assets/videos/dance-solo.mov"
    [obb_detection]="assets/videos/obb.mp4"
    [classification]="assets/videos/dogs.mp4"
    [instance_segmentation]="assets/videos/blackbox-city-road.mp4"
    [semantic_segmentation]="assets/videos/blackbox-city-road.mp4"
    [depth_estimation]="assets/videos/blackbox-city-road.mp4"
    [image_denoising]="assets/videos/noisy_hand.mp4"
    [super_resolution]="assets/videos/blackbox-city-road.mp4"
    [image_enhancement]="assets/videos/lowlight.mp4"
    [embedding]="assets/videos/face-pair-sofa.mp4"
    [attribute_recognition]="assets/videos/person-pair-hallway.mp4"
    [reid]="assets/videos/person-pair-hallway.mp4"
    [ppu]="assets/videos/blackbox-city-road.mp4"
    [hand_landmark]="assets/videos/hand.mp4"
    [face_alignment]="assets/videos/face-alignment-closeup.mp4"
)
CATEGORY_DISPLAY=(
    [object_detection]="Object Detection"
    [face_detection]="Face Detection"
    [pose_estimation]="Pose Estimation"
    [obb_detection]="OBB (Oriented Bounding Box)"
    [classification]="Classification"
    [instance_segmentation]="Instance Segmentation"
    [semantic_segmentation]="Semantic Segmentation"
    [depth_estimation]="Depth Estimation"
    [image_denoising]="Image Denoising"
    [super_resolution]="Super Resolution"
    [image_enhancement]="Image Enhancement"
    [embedding]="Embedding"
    [attribute_recognition]="Attribute Recognition"
    [reid]="Person Re-Identification"
    [ppu]="PPU (Post-Processing Unit)"
    [hand_landmark]="Hand Landmark"
    [face_alignment]="Face Alignment"
)

CATEGORY_ORDER=(
    object_detection face_detection pose_estimation obb_detection classification
    instance_segmentation semantic_segmentation depth_estimation
    image_denoising super_resolution image_enhancement
    embedding attribute_recognition reid ppu
    hand_landmark face_alignment
)

# Per-model input overrides (for PPU models whose actual task differs from category)
declare -A MODEL_IMAGE_OVERRIDE MODEL_VIDEO_OVERRIDE
MODEL_IMAGE_OVERRIDE=(
    [scrfd500m_ppu]="sample/img/sample_face.jpg"
    [yolov5pose_ppu]="sample/img/sample_people.jpg"
    [handlandmarklite_1]="sample/img/sample_hand.jpg"
    [unet_mobilenet_v2]="sample/img/sample_dog.jpg"
)
MODEL_VIDEO_OVERRIDE=(
    [scrfd500m_ppu]="assets/videos/dance-solo.mov"
    [yolov5pose_ppu]="assets/videos/dance-solo.mov"
    [handlandmarklite_1]="assets/videos/hand.mp4"
)

# Categories that only support image tests (no video/async)
IMAGE_ONLY_CATEGORIES="embedding reid attribute_recognition"

# ============================================================
# Interactive Mode (when no arguments provided)
# ============================================================
interactive_menu() {
    local CONFIG_FILE="${PROJECT_ROOT}/config/test_models.conf"

    echo ""
    printf "${CYAN}%s${NC}\n" "═══════════════════════════════════════════════════════════════"
    printf "  ${CYAN}DX-APP Example Runner — Interactive Mode${NC}\n"
    printf "  Select options step-by-step. Press Enter for defaults.\n"
    printf "${CYAN}%s${NC}\n" "═══════════════════════════════════════════════════════════════"

    # ── Stage 1/6: Language ──
    echo ""
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    printf "  ${CYAN}[ Stage 1 / 6 ]  Language Selection${NC}\n"
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    echo ""
    printf "   1: C++ only\n"
    printf "   2: Python only\n"
    printf "   3: Both (C++ + Python)\n"
    echo ""
    while true; do
        printf "  Select [1-3, default: 3]: "
        read -r _lang_sel
        [[ -z "$_lang_sel" ]] && _lang_sel="3"
        case "$_lang_sel" in
            1) LANG_MODE="cpp"; break ;;
            2) LANG_MODE="py"; break ;;
            3) LANG_MODE="both"; break ;;
            *) printf "${RED}  Invalid: '%s'. Enter 1, 2, or 3.${NC}\n" "$_lang_sel" ;;
        esac
    done

    # ── Stage 2/6: Category ──
    # Count models per category from config
    declare -A _cat_count
    if [[ -f "$CONFIG_FILE" ]]; then
        while IFS=$'\t' read -r _name _cat _file; do
            [[ -z "$_name" || "$_name" == \#* ]] && continue
            _cat_count["$_cat"]=$(( ${_cat_count["$_cat"]:-0} + 1 ))
        done < "$CONFIG_FILE"
    fi

    echo ""
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    printf "  ${CYAN}[ Stage 2 / 6 ]  Category Selection${NC}\n"
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    echo ""
    printf "   ${YELLOW} 0: ALL (all categories)${NC}\n"
    local _idx=1
    local -a _cat_list=()
    echo ""
    for _cat in "${CATEGORY_ORDER[@]}"; do
        _cat_list+=("$_cat")
        local _cnt="${_cat_count[$_cat]:-0}"
        printf "  %2d: %-30s (%d models)\n" "$_idx" "${CATEGORY_DISPLAY[$_cat]}" "$_cnt"
        _idx=$((_idx + 1))
    done
    local _cat_max=$(( ${#_cat_list[@]} ))
    echo ""
    while true; do
        printf "  Select [0-%d, default: 0]: " "$_cat_max"
        read -r _cat_sel
        [[ -z "$_cat_sel" ]] && _cat_sel="0"
        if [[ "$_cat_sel" =~ ^[0-9]+$ ]] && [ "$_cat_sel" -ge 0 ] && [ "$_cat_sel" -le "$_cat_max" ]; then
            break
        fi
        printf "${RED}  Invalid: '%s'. Enter a number between 0 and %d.${NC}\n" "$_cat_sel" "$_cat_max"
    done
    if [ "$_cat_sel" -eq 0 ]; then
        CATEGORY_FILTER=""
    else
        CATEGORY_FILTER="${_cat_list[$((_cat_sel - 1))]}"
        # Show models in selected category
        echo ""
        printf "  ${YELLOW}Models in ${CATEGORY_DISPLAY[$CATEGORY_FILTER]}:${NC}\n"
        local _col=0
        while IFS=$'\t' read -r _mn _mc _mf; do
            [[ -z "$_mn" || "$_mn" == \#* ]] && continue
            [[ "$_mc" != "$CATEGORY_FILTER" ]] && continue
            printf "  %-28s" "$_mn"
            _col=$((_col + 1))
            (( _col % 3 == 0 )) && echo ""
        done < "$CONFIG_FILE"
        (( _col % 3 != 0 )) && echo ""
    fi

    # ── Stage 3/6: Model Filter ──
    echo ""
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    printf "  ${CYAN}[ Stage 3 / 6 ]  Model Filter${NC}\n"
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    echo ""
    printf "   1: ALL (run all matching models)\n"
    printf "   2: Filter by keyword\n"
    echo ""
    while true; do
        printf "  Select [1-2, default: 1]: "
        read -r _filter_sel
        [[ -z "$_filter_sel" ]] && _filter_sel="1"
        case "$_filter_sel" in
            1) FILTER=""; break ;;
            2)
                printf "  Enter filter keyword (e.g. yolov5, resnet, scrfd, efficientnet): "
                read -r FILTER
                # Lowercase for consistent matching (model names are lowercase)
                FILTER="${FILTER,,}"
                break ;;
            *) printf "${RED}  Invalid: '%s'. Enter 1 or 2.${NC}\n" "$_filter_sel" ;;
        esac
    done

    # ── Stage 4/6: Execution Mode ──
    echo ""
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    printf "  ${CYAN}[ Stage 4 / 6 ]  Execution Mode${NC}\n"
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    echo ""
    printf "   1: sync + async\n"
    printf "   2: sync only\n"
    printf "   3: async only\n"
    echo ""
    while true; do
        printf "  Select [1-3, default: 1]: "
        read -r _mode_sel
        [[ -z "$_mode_sel" ]] && _mode_sel="1"
        case "$_mode_sel" in
            1) SKIP_SYNC=false; SKIP_ASYNC=false; break ;;
            2) SKIP_SYNC=false; SKIP_ASYNC=true; break ;;
            3) SKIP_SYNC=true; SKIP_ASYNC=false; break ;;
            *) printf "${RED}  Invalid: '%s'. Enter 1, 2, or 3.${NC}\n" "$_mode_sel" ;;
        esac
    done

    # ── Stage 5/6: Input Type ──
    echo ""
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    printf "  ${CYAN}[ Stage 5 / 6 ]  Input Type${NC}\n"
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    echo ""
    printf "   1: image + video\n"
    printf "   2: image only\n"
    printf "   3: video only\n"
    echo ""
    while true; do
        printf "  Select [1-3, default: 1]: "
        read -r _input_sel
        [[ -z "$_input_sel" ]] && _input_sel="1"
        case "$_input_sel" in
            1) SKIP_VIDEO=false; SKIP_IMAGE=false; break ;;
            2) SKIP_VIDEO=true; SKIP_IMAGE=false; break ;;
            3) SKIP_VIDEO=false; SKIP_IMAGE=true; break ;;
            *) printf "${RED}  Invalid: '%s'. Enter 1, 2, or 3.${NC}\n" "$_input_sel" ;;
        esac
    done

    # ── Stage 6/6: Additional Options ──
    echo ""
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    printf "  ${CYAN}[ Stage 6 / 6 ]  Additional Options${NC}\n"
    printf "${CYAN}%s${NC}\n" "───────────────────────────────────────────────────────────────"
    echo ""
    printf "  Enable GUI display? [y/N, default: N]: "
    read -r _display_sel
    if [[ "$_display_sel" == "y" || "$_display_sel" == "Y" ]]; then
        DISPLAY_ARG=""
        printf "  Display delay seconds [default: 1.0]: "
        read -r _delay_sel
        [[ -n "$_delay_sel" ]] && DISPLAY_DELAY="$_delay_sel"
    fi

    printf "  Save output images? [y/N, default: N]: "
    read -r _save_sel
    if [[ "$_save_sel" == "y" || "$_save_sel" == "Y" ]]; then
        SAVE_IMAGES=true
        SAVE_ARG="-s"
        PY_SAVE_ARG="--save"
        printf "  Save directory [default: logs/saved_images]: "
        read -r _save_dir_sel
        if [[ -n "$_save_dir_sel" ]]; then
            SAVE_DIR="$_save_dir_sel"
        else
            SAVE_DIR="logs/saved_images"
        fi
        SAVE_DIR_ARG="--save-dir ${SAVE_DIR}"
    fi

    # ── Confirmation ──
    local _lang_display="$LANG_MODE"
    local _cat_display="${CATEGORY_FILTER:-ALL}"
    local _filter_display="${FILTER:-ALL}"
    local _mode_display="sync + async"
    [[ "$SKIP_ASYNC" == true ]] && _mode_display="sync only"
    [[ "$SKIP_SYNC" == true ]] && _mode_display="async only"
    local _input_display="image + video"
    [[ "$SKIP_VIDEO" == true ]] && _input_display="image only"
    [[ "$SKIP_IMAGE" == true ]] && _input_display="video only"
    local _display_display="off"
    [[ -z "$DISPLAY_ARG" ]] && _display_display="on (delay: ${DISPLAY_DELAY}s)"
    local _save_display="off"
    [[ "$SAVE_IMAGES" == true ]] && _save_display="on (dir: ${SAVE_DIR})"

    echo ""
    printf "${CYAN}%s${NC}\n" "═══════════════════════════════════════════════════════════════"
    printf "  ${CYAN}Configuration Summary${NC}\n"
    printf "${CYAN}%s${NC}\n" "═══════════════════════════════════════════════════════════════"
    printf "  ${GREEN}Language${NC}  : %s\n" "$_lang_display"
    printf "  ${GREEN}Category${NC}  : %s\n" "$_cat_display"
    printf "  ${GREEN}Filter${NC}    : %s\n" "$_filter_display"
    printf "  ${GREEN}Mode${NC}      : %s\n" "$_mode_display"
    printf "  ${GREEN}Input${NC}     : %s\n" "$_input_display"
    printf "  ${GREEN}Display${NC}   : %s\n" "$_display_display"
    printf "  ${GREEN}Save${NC}      : %s\n" "$_save_display"
    printf "${CYAN}%s${NC}\n" "═══════════════════════════════════════════════════════════════"
    echo ""
    printf "  Proceed? [Y/n, default: Y]: "
    read -r _confirm
    if [[ "$_confirm" == "n" || "$_confirm" == "N" ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
}

# ============================================================
# Option Parsing
# ============================================================
LANG_MODE="both"
SKIP_VIDEO=false
SKIP_IMAGE=false
SKIP_SYNC=false
SKIP_ASYNC=false
DISPLAY_ARG="--no-display"
FILTER=""
CATEGORY_FILTER=""
EXE_FILTER=""
DISPLAY_DELAY=1.0
SLEEP_AFTER_TEST=1.0
SAVE_IMAGES=false
SAVE_ARG=""
PY_SAVE_ARG=""
SAVE_DIR=""
SAVE_DIR_ARG=""

# Enter interactive mode when no CLI arguments are provided
if [[ $# -eq 0 ]]; then
    interactive_menu
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --lang)
            LANG_MODE="$2"
            shift 2 ;;
        --no-video|--skip-video)
            SKIP_VIDEO=true
            shift ;;
        --video-only)
            SKIP_IMAGE=true
            shift ;;
        --display)
            DISPLAY_ARG=""
            shift ;;
        --display-delay)
            DISPLAY_DELAY="$2"
            shift 2 ;;
        --save-images)
            SAVE_IMAGES=true
            SAVE_ARG="-s"
            PY_SAVE_ARG="--save"
            shift ;;
        --save-dir)
            SAVE_DIR="$2"
            SAVE_DIR_ARG="--save-dir ${SAVE_DIR}"
            shift 2 ;;
        --exe)
            EXE_FILTER="$2"
            shift 2 ;;
        --no-display)
            DISPLAY_ARG="--no-display"
            shift ;;
        --sync-only)
            SKIP_ASYNC=true
            shift ;;
        --async-only)
            SKIP_SYNC=true
            shift ;;
        --filter)
            FILTER="$2"
            shift 2 ;;
        --category)
            CATEGORY_FILTER="$2"
            shift 2 ;;
        -h|--help)
            cat << 'HELPEOF'
Usage: scripts/run_examples.sh [OPTIONS]

Options:
  --lang <cpp|py|both>  Language selection (default: both)
  --display             Enable GUI display
  --no-display          Disable GUI display (default)
  --display-delay <s>   Seconds to wait between displayed tests (default: 1.0)
  --no-video            Skip video tests (image only)
  --skip-video          Alias for --no-video
  --video-only          Skip image tests (video only)
  --sync-only           Run sync tests only (C++ only)
  --async-only          Run async tests only (C++ only)
  --filter <STR>        Only run models whose name contains STR
  --category <CAT>      Only run models in the specified category
  --exe <NAME>          Run a specific C++ executable by basename (e.g. yolov9s_sync)
  --save-images         Save output images/frames for each test
  --save-dir <dir>      Base directory for saved outputs (passed through to examples)
  -h, --help            Show this help message

Environment Variables:
  BUILD_DIR=<dir>       Override the C++ build output directory (default: bin)

Examples:
  scripts/run_examples.sh --lang cpp --sync-only --no-video
  scripts/run_examples.sh --lang py --filter yolov9
  scripts/run_examples.sh --lang both --filter scrfd
  scripts/run_examples.sh --exe yolov9s_sync
  BUILD_DIR=build_x86_64/release/bin scripts/run_examples.sh --lang cpp
HELPEOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)"
            exit 1
            ;;
    esac
done

# Validate lang
case "$LANG_MODE" in
    cpp|py|both) ;;
    *) echo "Error: --lang must be cpp, py, or both (got: $LANG_MODE)"; exit 1 ;;
esac

# If display is enabled, use the display delay between tests
if [ "${DISPLAY_ARG}" = "" ]; then
    SLEEP_AFTER_TEST="${DISPLAY_DELAY}"
fi

# ============================================================
# Validate prerequisites
# ============================================================
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}Error:${NC} Model config not found: ${CONFIG_FILE}"
    exit 1
fi

if [[ "$LANG_MODE" == "cpp" || "$LANG_MODE" == "both" ]]; then
    if [ ! -d "${BUILD_DIR}" ]; then
        echo -e "${YELLOW}Warning:${NC} C++ build directory not found: ${BUILD_DIR}"
        if [[ "$LANG_MODE" == "cpp" ]]; then
            echo "Please build first:"
            echo "  cd src/cpp_example && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
            exit 1
        else
            echo "Skipping C++ tests."
            LANG_MODE="py"
        fi
    fi
fi

# ============================================================
# Load Model Registry
# ============================================================
declare -A MODEL_FILE MODEL_CATEGORY

while IFS=$'\t' read -r name category model_file; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    MODEL_FILE["$name"]="$model_file"
    MODEL_CATEGORY["$name"]="$category"
done < "${CONFIG_FILE}"

echo "Loaded ${#MODEL_FILE[@]} models from config" >&2

# ============================================================
# Logging Setup
# ============================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/test_${LANG_MODE}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SUMMARY_LOG="${LOG_DIR}/summary.log"
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# ============================================================
# Common Test Runner Functions
# ============================================================
run_test() {
    local test_name="$1"
    local cmd="$2"
    local log_file="${LOG_DIR}/${test_name}.log"
    local error_log="${LOG_DIR}/${test_name}.error.log"

    echo -e "${BLUE}[RUNNING]${NC} ${test_name}" | tee -a "${SUMMARY_LOG}"
    echo "Command: ${cmd}" >> "${SUMMARY_LOG}"

    if eval "${cmd}" > >(tee "${log_file}") 2> "${error_log}"; then
        echo -e "${GREEN}[SUCCESS]${NC} ${test_name}" | tee -a "${SUMMARY_LOG}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        [ ! -s "${error_log}" ] && rm -f "${error_log}"
    else
        local exit_code=$?
        echo -e "${RED}[FAILED]${NC} ${test_name} (Exit code: ${exit_code})" | tee -a "${SUMMARY_LOG}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "Exit code: ${exit_code}" >> "${error_log}"
    fi
    echo "" >> "${SUMMARY_LOG}"
    sleep "${SLEEP_AFTER_TEST}"
}

run_image_test() {
    if [ "$SKIP_IMAGE" = true ]; then
        echo -e "${YELLOW}[SKIP]${NC} $1 (--video-only mode)" | tee -a "${SUMMARY_LOG}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi
    run_test "$1" "$2"
}

run_video_test() {
    if [ "$SKIP_VIDEO" = true ]; then
        echo -e "${YELLOW}[SKIP]${NC} $1 (--no-video mode)" | tee -a "${SUMMARY_LOG}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi
    run_test "$1" "$2"
}

# ============================================================
# C++ Test Functions
# ============================================================
check_executable() {
    local exe_path="$1"
    local test_name="$2"
    if [ ! -f "${exe_path}" ]; then
        echo -e "${YELLOW}[SKIP]${NC} ${test_name} (Not built: ${exe_path})" | tee -a "${SUMMARY_LOG}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 1
    fi
    return 0
}

run_sync_test() {
    if [ "$SKIP_SYNC" = true ]; then
        echo -e "${YELLOW}[SKIP]${NC} $1 (--async-only mode)" | tee -a "${SUMMARY_LOG}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi
    local test_type="$3"
    if [ "$test_type" = "video" ]; then
        run_video_test "$1" "$2"
    else
        run_image_test "$1" "$2"
    fi
}

run_async_test() {
    if [ "$SKIP_ASYNC" = true ]; then
        echo -e "${YELLOW}[SKIP]${NC} $1 (--sync-only mode)" | tee -a "${SUMMARY_LOG}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi
    local test_type="$3"
    if [ "$test_type" = "video" ]; then
        run_video_test "$1" "$2"
    else
        run_image_test "$1" "$2"
    fi
}

# run_cpp_model <model_name> <model_file> <image> <video> <category>
run_cpp_model() {
    local model_name="$1"
    local model_file="$2"
    local image="$3"
    local video="$4"
    local category="$5"
    local sync_exe="${BUILD_DIR}/${model_name}_sync"
    local async_exe="${BUILD_DIR}/${model_name}_async"

    local is_image_only=false
    [[ -n "$category" && "$IMAGE_ONLY_CATEGORIES" == *"$category"* ]] && is_image_only=true

    echo "=== [C++] ${model_name} ===" | tee -a "${SUMMARY_LOG}"

    if check_executable "${sync_exe}" "cpp_${model_name}_sync_img"; then
        run_sync_test "cpp_${model_name}_sync_img" \
            "DXAPP_SAVE_IMAGE=\"${SAVE_DIR}/${model_name}_sync/$(basename ${image%/})_output.jpg\" ${sync_exe} -m \"${model_file}\" -i ${image} ${DISPLAY_ARG} ${SAVE_ARG} ${SAVE_DIR_ARG} -l -1" "image"
    fi
    if check_executable "${async_exe}" "cpp_${model_name}_async_img"; then
        run_async_test "cpp_${model_name}_async_img" \
            "DXAPP_SAVE_IMAGE=\"${SAVE_DIR}/${model_name}_async/$(basename ${image%/})_output.jpg\" ${async_exe} -m \"${model_file}\" -i ${image} ${DISPLAY_ARG} ${SAVE_ARG} ${SAVE_DIR_ARG} -l -1" "image"
    fi
    if ! $is_image_only; then
        if check_executable "${sync_exe}" "cpp_${model_name}_sync_video"; then
            run_sync_test "cpp_${model_name}_sync_video" \
                "${sync_exe} -m \"${model_file}\" -v ${video} ${DISPLAY_ARG} ${SAVE_ARG} ${SAVE_DIR_ARG} -l 1" "video"
        fi
        if check_executable "${async_exe}" "cpp_${model_name}_async_video"; then
            run_async_test "cpp_${model_name}_async_video" \
                "${async_exe} -m \"${model_file}\" -v ${video} ${DISPLAY_ARG} ${SAVE_ARG} ${SAVE_DIR_ARG} -l 1" "video"
        fi
    fi
}

# ============================================================
# Python Test Functions
# ============================================================
# run_py_model <model_name> <py_category_dir> <category> <model_file> <image> <video>
run_py_model() {
    local model_name="$1"
    local py_cat="$2"
    local category="$3"
    local model_file="$4"
    local image="$5"
    local video="$6"
    local model_dir="${PY_BASE}/${py_cat}/${model_name}"

    local is_image_only=false
    [[ "$IMAGE_ONLY_CATEGORIES" == *"$category"* ]] && is_image_only=true

    echo "=== [Python] ${model_name} ===" | tee -a "${SUMMARY_LOG}"

    # Discover available scripts
    local scripts=()
    while IFS= read -r -d '' f; do
        scripts+=("$f")
    done < <(find "$model_dir" -maxdepth 1 -name "${model_name}_*.py" -print0 | sort -z)

    if [ ${#scripts[@]} -eq 0 ]; then
        echo -e "${YELLOW}[SKIP]${NC} ${model_name}: no scripts found" | tee -a "${SUMMARY_LOG}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return
    fi

    for script in "${scripts[@]}"; do
        local basename_script=$(basename "$script" .py)
        local suffix="${basename_script#${model_name}_}"
        local test_name="py_${model_name}_${suffix}"

        if [[ "$suffix" == async* ]]; then
            run_image_test "${test_name}_img" \
                "python ${script} --model \"${model_file}\" --image ${image} ${DISPLAY_ARG} ${PY_SAVE_ARG} ${SAVE_DIR_ARG}"
            if ! $is_image_only; then
                run_video_test "${test_name}_video" \
                    "python ${script} --model \"${model_file}\" --video ${video} ${DISPLAY_ARG} ${PY_SAVE_ARG} ${SAVE_DIR_ARG}"
            fi

        elif [[ "$suffix" == "sync" ]]; then
            run_image_test "${test_name}_img" \
                "python ${script} --model \"${model_file}\" --image ${image} ${DISPLAY_ARG} ${PY_SAVE_ARG} ${SAVE_DIR_ARG}"
            if ! $is_image_only; then
                run_video_test "${test_name}_video" \
                    "python ${script} --model \"${model_file}\" --video ${video} ${DISPLAY_ARG} ${PY_SAVE_ARG} ${SAVE_DIR_ARG}"
            fi

        elif [[ "$suffix" == sync* ]]; then
            run_image_test "${test_name}_img" \
                "python ${script} --model \"${model_file}\" --image ${image} ${DISPLAY_ARG} ${PY_SAVE_ARG} ${SAVE_DIR_ARG}"
            if ! $is_image_only; then
                run_video_test "${test_name}_video" \
                    "python ${script} --model \"${model_file}\" --video ${video} ${DISPLAY_ARG} ${PY_SAVE_ARG} ${SAVE_DIR_ARG}"
            fi
        fi
    done

}

# ============================================================
# Discover models
# ============================================================
# C++ model discovery
declare -a CPP_MODELS=()
if [[ "$LANG_MODE" == "cpp" || "$LANG_MODE" == "both" ]]; then
    if [[ -n "${EXE_FILTER}" ]]; then
        # User requested an exact executable by basename
        if [ -f "${BUILD_DIR}/${EXE_FILTER}" ]; then
            model_name=$(echo "${EXE_FILTER}" | sed 's/_sync$//; s/_async$//')
            CPP_MODELS+=("${model_name}")
        fi
    else
        for sync_exe in "${BUILD_DIR}"/*_sync; do
            [ -f "$sync_exe" ] || continue
            model_name=$(basename "$sync_exe" | sed 's/_sync$//')
            if [[ -n "$FILTER" ]]; then
                _mn_lower="${model_name,,}"
                _fl_lower="${FILTER,,}"
                [[ "$_mn_lower" != *"$_fl_lower"* ]] && continue
            fi
            [[ -n "$CATEGORY_FILTER" && "${MODEL_CATEGORY[$model_name]}" != "$CATEGORY_FILTER" ]] && continue
            [[ "$model_name" == *"_x_"* ]] && continue
            CPP_MODELS+=("$model_name")
        done
    fi
fi

# Python model discovery
declare -a PY_MODELS=()  # "model_name|py_dir|category"
if [[ "$LANG_MODE" == "py" || "$LANG_MODE" == "both" ]]; then
    for category_dir in "${PY_BASE}"/*/; do
        py_cat=$(basename "$category_dir")
        [[ "$py_cat" == "common" || "$py_cat" == "__pycache__" || "$py_cat" == "utils" ]] && continue

        for model_dir in "${category_dir}"*/; do
            [ -d "$model_dir" ] || continue
            model_name=$(basename "$model_dir")
            [[ "$model_name" == "__pycache__" || "$model_name" == "factory" ]] && continue
            if [[ -n "$FILTER" ]]; then
                _mn_lower="${model_name,,}"
                _fl_lower="${FILTER,,}"
                [[ "$_mn_lower" != *"$_fl_lower"* ]] && continue
            fi

            category="${MODEL_CATEGORY[$model_name]}"
            [ -z "$category" ] && category="$py_cat"
            [[ -n "$CATEGORY_FILTER" && "$category" != "$CATEGORY_FILTER" ]] && continue
            PY_MODELS+=("${model_name}|${py_cat}|${category}")
        done
    done
fi

# ============================================================
# Header
# ============================================================
echo "========================================" | tee "${SUMMARY_LOG}"
echo "DX-APP Test Suite (lang: ${LANG_MODE})" | tee -a "${SUMMARY_LOG}"
echo "Started at: $(date)" | tee -a "${SUMMARY_LOG}"
echo "Config file: ${CONFIG_FILE}" | tee -a "${SUMMARY_LOG}"
echo "Log directory: ${LOG_DIR}" | tee -a "${SUMMARY_LOG}"

mode_desc=""
[ "$SKIP_VIDEO" = true ] && mode_desc="image-only"
[ "$SKIP_IMAGE" = true ] && mode_desc="video-only"
[ "$SKIP_SYNC" = true ] && mode_desc="async-only"
[ "$SKIP_ASYNC" = true ] && mode_desc="sync-only"
[ -z "$mode_desc" ] && mode_desc="full (sync+async, image+video)"
echo "Mode: ${mode_desc}" | tee -a "${SUMMARY_LOG}"
[ -n "$FILTER" ] && echo -e "${YELLOW}Filter: ${FILTER}${NC}" | tee -a "${SUMMARY_LOG}"

if [[ "$LANG_MODE" == "cpp" || "$LANG_MODE" == "both" ]]; then
    BUILT_COUNT=$(find "${BUILD_DIR}" -maxdepth 1 -type f -executable 2>/dev/null | wc -l)
    echo -e "${BLUE}C++ built executables: ${BUILT_COUNT}, discovered: ${#CPP_MODELS[@]}${NC}" | tee -a "${SUMMARY_LOG}"
fi
if [[ "$LANG_MODE" == "py" || "$LANG_MODE" == "both" ]]; then
    echo -e "${BLUE}Python discovered: ${#PY_MODELS[@]} models${NC}" | tee -a "${SUMMARY_LOG}"
fi
echo "========================================" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"

# ============================================================
# Run tests grouped by category
# ============================================================
CPP_UNCATEGORIZED=0
PY_UNCATEGORIZED=0

for category in "${CATEGORY_ORDER[@]}"; do
    image="${CATEGORY_IMAGE[$category]}"
    video="${CATEGORY_VIDEO[$category]}"

    # Collect C++ models for this category
    cpp_cat_models=()
    for model_name in "${CPP_MODELS[@]}"; do
        [[ "${MODEL_CATEGORY[$model_name]}" == "$category" ]] && cpp_cat_models+=("$model_name")
    done

    # Collect Python models for this category
    py_cat_models=()
    for entry in "${PY_MODELS[@]}"; do
        IFS='|' read -r m_name m_pydir m_cat <<< "$entry"
        [[ "$m_cat" == "$category" ]] && py_cat_models+=("$entry")
    done

    total_count=$(( ${#cpp_cat_models[@]} + ${#py_cat_models[@]} ))
    [ $total_count -eq 0 ] && continue

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "${SUMMARY_LOG}"
    echo -e "${BLUE}${CATEGORY_DISPLAY[$category]} (${total_count} models)${NC}" | tee -a "${SUMMARY_LOG}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "${SUMMARY_LOG}"
    echo "" | tee -a "${SUMMARY_LOG}"

    # C++ tests
    for model_name in "${cpp_cat_models[@]}"; do
        model_file="${MODEL_FILE[$model_name]}"
        if [ -z "$model_file" ]; then
            echo -e "${YELLOW}[WARN]${NC} C++ ${model_name}: no model file in config — skipping" | tee -a "${SUMMARY_LOG}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        if [ ! -f "$model_file" ]; then
            echo -e "${YELLOW}[WARN]${NC} C++ ${model_name}: model file not found (${model_file}) — skipping" | tee -a "${SUMMARY_LOG}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        m_image="${MODEL_IMAGE_OVERRIDE[$model_name]:-$image}"
        m_video="${MODEL_VIDEO_OVERRIDE[$model_name]:-$video}"
        run_cpp_model "$model_name" "$model_file" "$m_image" "$m_video" "$category"
    done

    # Python tests
    for entry in "${py_cat_models[@]}"; do
        IFS='|' read -r model_name py_cat _ <<< "$entry"
        model_file="${MODEL_FILE[$model_name]}"
        if [ -z "$model_file" ]; then
            echo -e "${YELLOW}[WARN]${NC} Python ${model_name}: no model file in config — skipping" | tee -a "${SUMMARY_LOG}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        if [ ! -f "$model_file" ]; then
            echo -e "${YELLOW}[WARN]${NC} Python ${model_name}: model file not found (${model_file}) — skipping" | tee -a "${SUMMARY_LOG}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        m_image="${MODEL_IMAGE_OVERRIDE[$model_name]:-$image}"
        m_video="${MODEL_VIDEO_OVERRIDE[$model_name]:-$video}"
        run_py_model "$model_name" "$py_cat" "$category" "$model_file" "$m_image" "$m_video"
    done
    echo "" | tee -a "${SUMMARY_LOG}"
done

# ============================================================
# Uncategorized models
# ============================================================
uncategorized_header_shown=false
show_uncat_header() {
    if ! $uncategorized_header_shown; then
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "${SUMMARY_LOG}"
        echo -e "${YELLOW}Uncategorized Models (not in config)${NC}" | tee -a "${SUMMARY_LOG}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "${SUMMARY_LOG}"
        uncategorized_header_shown=true
    fi
}

for model_name in "${CPP_MODELS[@]}"; do
    if [ -z "${MODEL_CATEGORY[$model_name]}" ]; then
        show_uncat_header
        echo -e "${YELLOW}[WARN]${NC} C++ ${model_name} (cpp_${model_name}): built but not in test_models.conf" | tee -a "${SUMMARY_LOG}"
        CPP_UNCATEGORIZED=$((CPP_UNCATEGORIZED + 1))
    fi
done

for entry in "${PY_MODELS[@]}"; do
    IFS='|' read -r model_name py_cat _ <<< "$entry"
    if [ -z "${MODEL_FILE[$model_name]}" ] && [ -z "${MODEL_CATEGORY[$model_name]}" ]; then
        show_uncat_header
        echo -e "${YELLOW}[WARN]${NC} Python ${model_name} (${py_cat}): not in test_models.conf" | tee -a "${SUMMARY_LOG}"
        PY_UNCATEGORIZED=$((PY_UNCATEGORIZED + 1))
    fi
done

# ============================================================
# Final Summary
# ============================================================
echo "" | tee -a "${SUMMARY_LOG}"
echo "========================================" | tee -a "${SUMMARY_LOG}"
echo "Test Suite Completed at: $(date)" | tee -a "${SUMMARY_LOG}"
echo "========================================" | tee -a "${SUMMARY_LOG}"
echo "" | tee -a "${SUMMARY_LOG}"
TOTAL=$((SUCCESS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo -e "Total: ${TOTAL}" | tee -a "${SUMMARY_LOG}"
echo -e "${GREEN}Successful:${NC} ${SUCCESS_COUNT}" | tee -a "${SUMMARY_LOG}"
echo -e "${RED}Failed:${NC} ${FAIL_COUNT}" | tee -a "${SUMMARY_LOG}"
echo -e "${YELLOW}Skipped:${NC} ${SKIP_COUNT}" | tee -a "${SUMMARY_LOG}"
TOTAL_UNCAT=$((CPP_UNCATEGORIZED + PY_UNCATEGORIZED))
if [ $TOTAL_UNCAT -gt 0 ]; then
    echo -e "${YELLOW}Uncategorized:${NC} ${TOTAL_UNCAT}" | tee -a "${SUMMARY_LOG}"
fi
echo "" | tee -a "${SUMMARY_LOG}"
echo "Full logs saved to: ${LOG_DIR}" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "" | tee -a "${SUMMARY_LOG}"
    echo -e "${RED}Failed Tests:${NC}" | tee -a "${SUMMARY_LOG}"
    grep -E "\[FAILED\]" "${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}_failures.txt"
fi

exit 0
