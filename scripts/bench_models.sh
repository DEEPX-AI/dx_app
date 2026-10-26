#!/bin/bash
# =============================================================================
# bench_models.sh - DX-APP Model Benchmark Script
# =============================================================================
# Automatically measures per-model inference speed and reports results as
# CSV + summary table. Targets models registered in config/test_models.conf,
# running C++/Python sync runners in --no-display mode and parsing the
# PERFORMANCE SUMMARY output.
#
# Model layout (v3.3.0, dx-modelzoo task/family/variant):
#   config/test_models.conf  family<TAB>task<TAB>model_file<TAB>variant
#   C++    : ${BUILD_DIR}/<variant>_sync
#   Python : src/python_example/<task>/<family>/<variant>/<variant>_sync.py
# One benchmark entry per variant (= .dxnn stem), so every registered model
# gets one CSV row per language.
#
# Metrics:
#   - Inference Avg Latency (ms)   : Pure inference average time
#   - Inference Throughput (FPS)   : Inference-only throughput
#   - Preprocess Avg Latency (ms)  : Preprocessing average time
#   - Postprocess Avg Latency (ms) : Postprocessing average time
#   - Overall FPS                  : Full pipeline throughput
#   - Total Frames                 : Total frames processed
#
# Usage:
#   scripts/bench_models.sh                                  # All (cpp+py)
#   scripts/bench_models.sh --lang cpp                       # C++ only
#   scripts/bench_models.sh --lang py                        # Python only
#   scripts/bench_models.sh --loops 5                        # 5 image iterations
#   scripts/bench_models.sh --warmup 2                       # 2 warmup runs
#   scripts/bench_models.sh --filter yolov8                  # yolov8 only
#   scripts/bench_models.sh --task image_classification      # One task only
#   scripts/bench_models.sh --video                          # Video benchmark
#   scripts/bench_models.sh --output-dir results/bench       # Custom output dir
#   scripts/bench_models.sh --csv-only                       # CSV only (skip table)
#   scripts/bench_models.sh --compare logs/bench_prev/results.csv  # Compare
#   scripts/bench_models.sh --help                           # Help
#
# Output:
#   logs/bench_YYYYMMDD_HHMMSS/
#     ├── results.csv           Full results CSV
#     ├── summary.txt           Summary table (also printed to terminal)
#     ├── <variant>_<lang>.log  Per-model execution log
#     └── errors.log            Error summary (created only when errors exist)
# =============================================================================

set +e  # continue on error

# ============================================================================
# Resolve paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# BUILD_DIR can be provided from the environment (e.g. BUILD_DIR=bin ./scripts/bench_models.sh)
if [ -z "${BUILD_DIR:-}" ]; then
    # bin/ is the install location from build.sh (docs refer to bin/*_sync). Prefer it over a build tree.
    CANDIDATE_DIRS=(
        "bin"
        "build_x86_64/release/bin"
        "build_x86_64/bin"
        "build_aarch64/release/bin"
        "build/release/bin"
        "build/bin"
        "src/cpp_example/build"
    )
    BUILD_DIR=""
    # 1) Pick the first candidate that actually contains a *_sync binary
    #    (an empty or partial build tree must not hide a complete bin/).
    for d in "${CANDIDATE_DIRS[@]}"; do
        if [ -d "${PROJECT_ROOT}/${d}" ] && ls "${PROJECT_ROOT}/${d}"/*_sync >/dev/null 2>&1; then
            BUILD_DIR="${d}"
            break
        fi
    done
    # 2) Otherwise use the first directory that exists (later steps SKIP per binary).
    if [ -z "${BUILD_DIR}" ]; then
        for d in "${CANDIDATE_DIRS[@]}"; do
            [ -d "${PROJECT_ROOT}/${d}" ] && { BUILD_DIR="${d}"; break; }
        done
    fi
    [ -z "${BUILD_DIR}" ] && BUILD_DIR="bin"
fi
PY_BASE="src/python_example"
CONFIG_FILE="${PROJECT_ROOT}/config/test_models.conf"

# Python interpreter for the examples (and for result parsing). PYTHON can be
# provided from the environment. Otherwise prefer an interpreter that can import
# dx_engine: the active venv, then the project venvs, then python3 on PATH.
if [ -z "${PYTHON:-}" ]; then
    PY_CANDIDATES=()
    [ -n "${VIRTUAL_ENV:-}" ] && PY_CANDIDATES+=("${VIRTUAL_ENV}/bin/python")
    PY_CANDIDATES+=(
        "${PROJECT_ROOT}/.venv/bin/python"
        "${PROJECT_ROOT}/venv/bin/python"
        "${PROJECT_ROOT}/../venv-dx-runtime/bin/python"
    )
    command -v python3 &>/dev/null && PY_CANDIDATES+=("$(command -v python3)")
    command -v python  &>/dev/null && PY_CANDIDATES+=("$(command -v python)")
    for p in "${PY_CANDIDATES[@]}"; do
        if [ -x "$p" ] && "$p" -c "import dx_engine" &>/dev/null; then
            PYTHON="$p"
            break
        fi
    done
    # No candidate has dx_engine: keep the first usable one for result parsing.
    if [ -z "${PYTHON:-}" ]; then
        for p in "${PY_CANDIDATES[@]}"; do
            [ -x "$p" ] && { PYTHON="$p"; break; }
        done
    fi
fi

# Per-run wall-clock limit in seconds (0 = none). A hung example must not stall
# the whole sweep.
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"

# Colors
RED='\033[0;31m';    GREEN='\033[0;32m';  YELLOW='\033[1;33m'
BLUE='\033[0;34m';   CYAN='\033[0;36m';   MAGENTA='\033[0;35m'
BOLD='\033[1m';      DIM='\033[2m';       NC='\033[0m'

# ============================================================================
# Category → Default Input Mapping (same as run_examples.sh)
# ============================================================================
# Keys are the task column of config/test_models.conf (dx-modelzoo task names);
# the legacy pre-v3.3.0 names are kept for configs that still use them.
declare -A CATEGORY_IMAGE CATEGORY_VIDEO CATEGORY_DISPLAY
CATEGORY_IMAGE=(
    [object_detection]="sample/img/sample_street.jpg"
    [face_detection]="sample/img/sample_face.jpg"
    [pose_estimation]="sample/img/sample_people.jpg"
    [obb_detection]="sample/img/sample_airport_satellite_view.png"
    [classification]="sample/img/sample_dog.jpg"
    [instance_segmentation]="sample/img/sample_street.jpg"
    [semantic_segmentation]="sample/img/sample_parking.jpg"
    [depth_estimation]="sample/img/sample_horse.jpg"
    [image_denoising]="sample/img/sample_denoising.jpg"
    [super_resolution]="sample/img/sample_lowres275x150.png"
    [image_enhancement]="sample/img/sample_lowlight.jpg"
    [embedding]="sample/img/face_pair"
    [attribute_recognition]="sample/img/sample_person_a1.jpg"
    [reid]="sample/img/person_pair"
    [ppu]="sample/img/sample_street.jpg"
    [hand_landmark]="sample/img/sample_hand.jpg"
    [hand_detection]="sample/img/sample_hand.jpg"
    [face_alignment]="sample/img/sample_face_a1.jpg"
    [keypoint_detection]="sample/img/sample_street.jpg"
    [object_pose_estimation]="sample/dope/000000.png"
    [panoptic_driving_perception]="sample/img/sample_parking.jpg"
    [3d_object_detection]="sample/kitti/velodyne/000049.bin"
    [image_classification]="sample/img/sample_dog.jpg"
    [oriented_object_detection]="sample/img/sample_airport_satellite_view.png"
    [face_recognition]="sample/img/face_pair"
    [face_attribute]="sample/img/sample_person_a1.jpg"
    [face_landmark]="sample/img/sample_face_a1.jpg"
    [person_attribute]="sample/img/sample_person_a1.jpg"
    [person_reid]="sample/reid/queries/sample_person_a2.jpg"
    [image_retrieval]="sample/img/sample_person_a2.jpg"
    [visual_place_recognition]="sample/vpr/queries/q1.jpg"
    [image_matting]="sample/img/sample_person_b.jpg"
    [low_light_enhancement]="sample/img/sample_lowlight.jpg"
    [mainbody_detection]="sample/img/sample_kitchen.jpg"
    [zero_shot_image_classification]="sample/img/sample_dog.jpg"
    [zero_shot_instance_segmentation]="sample/img/sample_street.jpg"
    [anomaly_detection]="sample/img/sample_parking.jpg"
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
    [super_resolution]="assets/videos/lowres-drone-city-road.mp4"
    [image_enhancement]="assets/videos/lowlight.mp4"
    [embedding]="assets/videos/face-pair-sofa.mp4"
    [attribute_recognition]="assets/videos/person-pair-hallway.mp4"
    [reid]="assets/videos/person-pair-hallway.mp4"
    [ppu]="assets/videos/blackbox-city-road.mp4"
    [hand_landmark]="assets/videos/hand.mp4"
    [hand_detection]="assets/videos/hand.mp4"
    [face_alignment]="assets/videos/face-alignment-closeup.mp4"
    [keypoint_detection]="assets/videos/snowboard.mp4"
    [object_pose_estimation]="assets/videos/snowboard.mp4"
    [panoptic_driving_perception]="assets/videos/blackbox-city-road.mp4"
    [3d_object_detection]="assets/videos/blackbox-city-road.mp4"
    [image_classification]="assets/videos/dogs.mp4"
    [oriented_object_detection]="assets/videos/obb.mp4"
    [face_landmark]="assets/videos/face-alignment-closeup.mp4"
    [image_matting]="assets/videos/person-pair-hallway.mp4"
    [low_light_enhancement]="assets/videos/lowlight.mp4"
    [mainbody_detection]="assets/videos/blackbox-city-road.mp4"
    [zero_shot_image_classification]="assets/videos/dogs.mp4"
    [zero_shot_instance_segmentation]="assets/videos/blackbox-city-road.mp4"
    [anomaly_detection]="assets/videos/blackbox-city-road.mp4"
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
    [hand_detection]="Hand Detection"
    [face_alignment]="Face Alignment"
    [keypoint_detection]="Keypoint Detection"
    [object_pose_estimation]="Object Pose Estimation"
    [panoptic_driving_perception]="Panoptic Driving Perception"
    [3d_object_detection]="3D Object Detection"
    [image_classification]="Image Classification"
    [oriented_object_detection]="Oriented Object Detection"
    [face_recognition]="Face Recognition"
    [face_attribute]="Face Attribute"
    [face_landmark]="Face Landmark"
    [person_attribute]="Person Attribute"
    [person_reid]="Person Re-Identification"
    [image_retrieval]="Image Retrieval"
    [visual_place_recognition]="Visual Place Recognition"
    [image_matting]="Image Matting"
    [low_light_enhancement]="Low Light Enhancement"
    [mainbody_detection]="Mainbody Detection"
    [zero_shot_image_classification]="Zero-shot Image Classification"
    [zero_shot_instance_segmentation]="Zero-shot Instance Segmentation"
    [anomaly_detection]="Anomaly Detection"
)

# Run order. A task missing here is still benchmarked, after these.
CATEGORY_ORDER=(
    object_detection face_detection pose_estimation obb_detection
    oriented_object_detection classification image_classification
    instance_segmentation semantic_segmentation depth_estimation
    image_denoising super_resolution image_enhancement low_light_enhancement
    embedding face_recognition attribute_recognition person_attribute
    face_attribute face_landmark reid person_reid image_retrieval
    visual_place_recognition image_matting mainbody_detection ppu
    hand_landmark hand_detection face_alignment
    keypoint_detection object_pose_estimation panoptic_driving_perception
    3d_object_detection anomaly_detection
    zero_shot_image_classification zero_shot_instance_segmentation
)

# Per-variant image overrides. Super-resolution input size depends on the scale
# factor: ESPCN upscales a 275x150 crop (task default), Real-ESRGAN a smaller
# 165x90 one so the x8 output stays a sane size.
declare -A MODEL_IMAGE_OVERRIDE=(
    [unet_mobilenetv2_256x256]="sample/img/sample_dog.jpg"
    [realesrgan-x2_192x192]="sample/img/sample_lowres165x90.png"
    [realesrgan-x4_192x192]="sample/img/sample_lowres165x90.png"
    [realesrgan-x8_192x192]="sample/img/sample_lowres165x90.png"
)

# Tasks whose examples take single images only (no video/stream). With --video
# they are benchmarked on their image input instead. Same set as
# run_examples.sh IMAGE_ONLY_CATEGORIES.
IMAGE_ONLY_CATEGORIES="embedding reid attribute_recognition face_recognition person_attribute face_attribute person_reid image_retrieval visual_place_recognition object_pose_estimation 3d_object_detection super_resolution anomaly_detection"

is_image_only_category() {
    [[ " ${IMAGE_ONLY_CATEGORIES} " == *" $1 "* ]]
}

# assets/models first, then workspace/res/models walking up from dx_app
# (same lookup as run_examples.sh).
resolve_dxnn() {
    local raw="$1"
    local name
    name=$(basename "$raw")
    if [[ -f "$raw" ]]; then
        echo "$raw"
        return
    fi
    if [[ -f "${PROJECT_ROOT}/assets/models/${name}" ]]; then
        echo "assets/models/${name}"
        return
    fi
    local dir="$PROJECT_ROOT"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "${dir}/workspace/res/models/${name}" ]]; then
            echo "${dir}/workspace/res/models/${name}"
            return
        fi
        dir=$(dirname "$dir")
    done
    echo "$raw"
}

# ============================================================================
# Option Parsing
# ============================================================================
LANG_MODE="both"
LOOPS=3
WARMUP=1
FILTER=""
CATEGORY_FILTER=""
USE_VIDEO=false
OUTPUT_DIR=""
CSV_ONLY=false
COMPARE_CSV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --lang)         LANG_MODE="$2";     shift 2 ;;
        --loops|-l)     LOOPS="$2";         shift 2 ;;
        --warmup|-w)    WARMUP="$2";        shift 2 ;;
        --filter|-f)    FILTER="$2";        shift 2 ;;
        --category|--task) CATEGORY_FILTER="$2"; shift 2 ;;
        --timeout)      BENCH_TIMEOUT="$2"; shift 2 ;;
        --video)        USE_VIDEO=true;     shift ;;
        --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --csv-only)     CSV_ONLY=true;      shift ;;
        --compare)      COMPARE_CSV="$2";   shift 2 ;;
        -h|--help)
            cat << 'HELPEOF'
Usage: scripts/bench_models.sh [OPTIONS]

Options:
  --lang <cpp|py|both>    Language selection (default: both)
  --loops, -l <N>         Loop iterations for image benchmark (default: 3)
  --warmup, -w <N>        Warmup runs before measurement (default: 1)
  --filter, -f <STR>      Only benchmark models whose variant or family contains
                          STR (case-insensitive)
  --category, --task <T>  Only benchmark models of task T (test_models.conf
                          column 2, e.g. image_classification)
  --timeout <SEC>         Per-run time limit, 0 = none (default: 900, env
                          BENCH_TIMEOUT)
  --video                 Use video input instead of image (longer test;
                          image-only tasks still use their image)
  --output-dir <DIR>      Custom output directory (default: logs/bench_TIMESTAMP)
  --csv-only              Output CSV only (skip summary table)
  --compare <CSV>         Compare with previous results CSV
  -h, --help              Show this help

Output:
  logs/bench_YYYYMMDD_HHMMSS/
    results.csv           Full results in CSV format
    summary.txt           Human-readable summary table
    <variant>_<lang>.log  Per-model execution log
    errors.log            Errors (if any)

Environment:
  BUILD_DIR               C++ binary directory (default: bin, else a build tree)
  PYTHON                  Python interpreter (default: first of $VIRTUAL_ENV,
                          .venv, venv, python3 that can import dx_engine)
  BENCH_TIMEOUT           Same as --timeout

CSV status values:
  OK, EXEC_ERROR, PARSE_ERROR, TIMEOUT, SKIP (no binary/script),
  NO_MODEL (.dxnn not downloaded), NO_INPUT (sample input missing)

Performance Summary Columns:
  Model          : Variant name = .dxnn stem (e.g., yolov8-n_640x640)
  Lang           : cpp or py
  Category       : Task category
  Infer (ms)     : Average inference latency (lower is better)
  Infer FPS      : Inference throughput (higher is better)
  Pre (ms)       : Average preprocessing latency
  Post (ms)      : Average postprocessing latency
  Overall FPS    : End-to-end throughput including I/O
  Frames         : Total frames processed

Examples:
  scripts/bench_models.sh --lang cpp --loops 5
  scripts/bench_models.sh --filter yolov8 --loops 10
  scripts/bench_models.sh --task image_classification --lang cpp
  scripts/bench_models.sh --lang py --video
  scripts/bench_models.sh --compare logs/bench_20250101_120000/results.csv
HELPEOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (try --help)"
            exit 1
            ;;
    esac
done

# Validate
case "$LANG_MODE" in
    cpp|py|both) ;;
    *) echo "Error: --lang must be cpp, py, or both (got: $LANG_MODE)"; exit 1 ;;
esac

if ! [[ "$LOOPS" =~ ^[0-9]+$ ]] || (( LOOPS < 1 )); then
    echo "Error: --loops must be a positive integer (got: $LOOPS)"; exit 1
fi
if ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
    echo "Error: --warmup must be a non-negative integer (got: $WARMUP)"; exit 1
fi
if ! [[ "$BENCH_TIMEOUT" =~ ^[0-9]+$ ]]; then
    echo "Error: --timeout must be a non-negative integer (got: $BENCH_TIMEOUT)"; exit 1
fi

# Wrap each run in timeout(1) when it is available.
TIMEOUT_CMD=()
if (( BENCH_TIMEOUT > 0 )) && command -v timeout &>/dev/null; then
    TIMEOUT_CMD=(timeout --kill-after=10 "${BENCH_TIMEOUT}")
fi

# ============================================================================
# Validate prerequisites
# ============================================================================
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
            echo "Skipping C++ benchmarks."
            LANG_MODE="py"
        fi
    fi
fi

if [ -z "${PYTHON:-}" ] || ! "$PYTHON" -c "import sys" &>/dev/null; then
    echo -e "${RED}Error:${NC} python3 is required for result parsing (set PYTHON=<interpreter>)"
    exit 1
fi
if [[ "$LANG_MODE" == "py" || "$LANG_MODE" == "both" ]] && ! "$PYTHON" -c "import dx_engine" &>/dev/null; then
    echo -e "${YELLOW}Warning:${NC} ${PYTHON} cannot import dx_engine — Python runs will fail."
    echo "  Activate the dx_app venv or set PYTHON=<venv>/bin/python."
fi

# ============================================================================
# Load Model Registry
# ============================================================================
# config/test_models.conf: family<TAB>task<TAB>model_file<TAB>variant.
# One entry per variant (the .dxnn stem, which is also the C++ executable and
# Python script basename). A pre-v3.3.0 3-column config has no column 4; its
# column 1 is then the executable basename (same rule as run_examples.sh).
declare -A MODEL_FILE MODEL_CATEGORY MODEL_FAMILY

while IFS=$'\t' read -r family category model_file variant _rest; do
    [[ -z "$family" || "$family" == \#* ]] && continue
    model_file="${model_file%$'\r'}"
    variant="${variant%$'\r'}"
    [[ -z "$variant" ]] && variant="$family"
    MODEL_FILE["$variant"]=$(resolve_dxnn "$model_file")
    MODEL_CATEGORY["$variant"]="$category"
    MODEL_FAMILY["$variant"]="$family"
done < "${CONFIG_FILE}"

echo -e "${DIM}Loaded ${#MODEL_FILE[@]} models from config${NC}" >&2

# ============================================================================
# Output Setup
# ============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ -z "$OUTPUT_DIR" ]]; then
    LOG_DIR="logs/bench_${TIMESTAMP}"
else
    LOG_DIR="$OUTPUT_DIR"
fi
mkdir -p "${LOG_DIR}"

CSV_FILE="${LOG_DIR}/results.csv"
SUMMARY_FILE="${LOG_DIR}/summary.txt"
ERROR_LOG="${LOG_DIR}/errors.log"

# CSV header
echo "model,lang,category,infer_ms,infer_fps,pre_ms,post_ms,overall_fps,frames,status" > "$CSV_FILE"

# Counters
BENCH_TOTAL=0
BENCH_OK=0
BENCH_FAIL=0
BENCH_SKIP=0

# ============================================================================
# parse_performance_summary — Extract metrics from PERFORMANCE SUMMARY block
# ============================================================================
# Input: runner log file
# Output: "infer_ms|infer_fps|pre_ms|post_ms|overall_fps|frames" (pipe-separated)
# ============================================================================
parse_performance_summary() {
    local log_file="$1"
    "$PYTHON" -c "
import re, sys

text = open(sys.argv[1], encoding='utf-8', errors='replace').read()

# ── C++ format: PERFORMANCE SUMMARY ────────────────────────────────────────
if 'PERFORMANCE SUMMARY' in text:
    blocks = text.split('PERFORMANCE SUMMARY')
    block = blocks[-1]  # last block

    def extract_pipeline(name):
        pat = re.compile(rf'\s*{name}\s+([\d.]+)\s+ms\s+([\d.]+)\s+FPS', re.IGNORECASE)
        m = pat.search(block)
        if m:
            return float(m.group(1)), float(m.group(2))
        return None, None

    pre_ms,  _          = extract_pipeline('Preprocess')
    infer_ms, infer_fps = extract_pipeline('Inference')
    post_ms, _          = extract_pipeline('Postprocess')

    m_overall = re.search(r'Overall\s+FPS\s*:\s*([\d.]+)', block)
    overall_fps = float(m_overall.group(1)) if m_overall else None

    m_frames = re.search(r'Total\s+Frames\s*:\s*(\d+)', block)
    frames = int(m_frames.group(1)) if m_frames else None

# ── Python format: IMAGE PROCESSING SUMMARY (1 block per run) ──────────────
elif 'IMAGE PROCESSING SUMMARY' in text:
    # Parse all summary blocks and average them
    sections = re.split(r'IMAGE PROCESSING SUMMARY', text)
    sections = sections[1:]  # skip text before first block

    pre_vals, infer_vals, post_vals, total_vals = [], [], [], []
    for sec in sections:
        m = re.search(r'Preprocess\s+([\d.]+)\s+ms', sec)
        if m: pre_vals.append(float(m.group(1)))
        m = re.search(r'Inference\s+([\d.]+)\s+ms', sec)
        if m: infer_vals.append(float(m.group(1)))
        m = re.search(r'Postprocess\s+([\d.]+)\s+ms', sec)
        if m: post_vals.append(float(m.group(1)))
        m = re.search(r'Total\s+Time\s*:\s*([\d.]+)\s+ms', sec)
        if m: total_vals.append(float(m.group(1)))

    if not infer_vals:
        print('|||||||')
        sys.exit(0)

    infer_ms    = sum(infer_vals) / len(infer_vals)
    pre_ms      = sum(pre_vals) / len(pre_vals)   if pre_vals   else None
    post_ms     = sum(post_vals) / len(post_vals) if post_vals  else None
    avg_total   = sum(total_vals) / len(total_vals) if total_vals else infer_ms
    infer_fps   = 1000.0 / infer_ms  if infer_ms  else None
    overall_fps = 1000.0 / avg_total if avg_total else None
    frames      = len(infer_vals)

else:
    print('|||||||')
    sys.exit(0)

parts = [
    f'{infer_ms:.2f}' if infer_ms is not None else '',
    f'{infer_fps:.1f}' if infer_fps is not None else '',
    f'{pre_ms:.2f}' if pre_ms is not None else '',
    f'{post_ms:.2f}' if post_ms is not None else '',
    f'{overall_fps:.1f}' if overall_fps is not None else '',
    f'{frames}' if frames is not None else '',
]
print('|'.join(parts))
" "$log_file" 2>/dev/null
}

# csv_row <model> <lang> <category> <status> [metrics "infer_ms|infer_fps|pre_ms|post_ms|overall_fps|frames"]
csv_row() {
    local metrics="${5:-|||||0}"
    echo "$1,$2,$3,${metrics//|/,},$4" >> "$CSV_FILE"
}

# record_failure <model> <lang> <category> <rc> <log_file>
# Classifies a non-zero exit (timeout(1) returns 124, or 137 after the kill).
record_failure() {
    local model_name="$1" lang="$2" category="$3" rc="$4" log_file="$5"
    if [[ ${#TIMEOUT_CMD[@]} -gt 0 && ( "$rc" -eq 124 || "$rc" -eq 137 ) ]]; then
        csv_row "$model_name" "$lang" "$category" TIMEOUT
        echo -e "    ${RED}✗${NC} Timed out after ${BENCH_TIMEOUT}s"
        echo "[TIMEOUT] ${model_name} ${lang} — exceeded ${BENCH_TIMEOUT}s (log: ${log_file})" >> "$ERROR_LOG"
    else
        csv_row "$model_name" "$lang" "$category" EXEC_ERROR
        echo -e "    ${RED}✗${NC} Execution failed (exit: ${rc})"
        echo "[EXEC_ERROR] ${model_name} ${lang} — exit code ${rc} (log: ${log_file})" >> "$ERROR_LOG"
    fi
    BENCH_FAIL=$((BENCH_FAIL + 1))
}

# record_result <model> <lang> <category> <log_file>
record_result() {
    local model_name="$1" lang="$2" category="$3" log_file="$4"
    local result
    result=$(parse_performance_summary "$log_file")
    if [[ -n "$result" && "$result" != "|||||||" ]]; then
        local infer_ms infer_fps pre_ms post_ms overall_fps frames
        IFS='|' read -r infer_ms infer_fps pre_ms post_ms overall_fps frames <<< "$result"
        csv_row "$model_name" "$lang" "$category" OK "$result"
        echo -e "    ${GREEN}✓${NC} Infer: ${CYAN}${infer_ms}${NC} ms (${CYAN}${infer_fps}${NC} FPS)  Overall: ${CYAN}${overall_fps}${NC} FPS  [${frames} frames]"
        BENCH_OK=$((BENCH_OK + 1))
    else
        csv_row "$model_name" "$lang" "$category" PARSE_ERROR
        echo -e "    ${YELLOW}⚠${NC} Failed to parse performance data"
        echo "[PARSE_ERROR] ${model_name} ${lang} — no PERFORMANCE SUMMARY found (log: ${log_file})" >> "$ERROR_LOG"
        BENCH_FAIL=$((BENCH_FAIL + 1))
    fi
}

# ============================================================================
# run_bench_cpp — Benchmark a single C++ model
# ============================================================================
# run_bench_cpp <variant> <model_file> <category> <input_flag> <input_path>
run_bench_cpp() {
    local model_name="$1"
    local model_file="$2"
    local category="$3"
    local input_flag="$4"
    local input_path="$5"
    local sync_exe="${BUILD_DIR}/${model_name}_sync"

    if [ ! -f "${sync_exe}" ]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (no sync binary: ${sync_exe})"
        csv_row "$model_name" cpp "$category" SKIP
        BENCH_SKIP=$((BENCH_SKIP + 1))
        return
    fi

    local log_file="${LOG_DIR}/${model_name}_cpp.log"
    local cmd=("${TIMEOUT_CMD[@]}" "${sync_exe}" -m "${model_file}" "${input_flag}" "${input_path}" --no-display)
    BENCH_TOTAL=$((BENCH_TOTAL + 1))

    # Warmup
    if (( WARMUP > 0 )); then
        echo -e "  ${DIM}[WarmUp]${NC} ${model_name} (cpp, ${WARMUP} runs)..."
        "${cmd[@]}" -l "${WARMUP}" > /dev/null 2>&1 || true
    fi

    # Actual benchmark
    echo -e "  ${BLUE}[Bench]${NC} ${model_name} (cpp, ${LOOPS} loops)..."
    "${cmd[@]}" -l "${LOOPS}" > "${log_file}" 2>&1
    local rc=$?
    if (( rc == 0 )); then
        record_result "$model_name" cpp "$category" "$log_file"
    else
        record_failure "$model_name" cpp "$category" "$rc" "$log_file"
    fi
}

# ============================================================================
# run_bench_py — Benchmark a single Python model
# ============================================================================
# run_bench_py <variant> <model_file> <category> <family> <input_flag> <input_path>
run_bench_py() {
    local model_name="$1"
    local model_file="$2"
    local category="$3"
    local family="$4"
    local input_flag="$5"
    local input_path="$6"

    # Sync script: <task>/<family>/<variant>/<variant>_sync.py. Fall back to a
    # search over all tasks when the config task and the directory disagree.
    local sync_script="${PY_BASE}/${category}/${family}/${model_name}/${model_name}_sync.py"
    if [[ ! -f "$sync_script" ]]; then
        sync_script=$(find "${PY_BASE}" -mindepth 4 -maxdepth 4 -type f \
            -path "*/${model_name}/${model_name}_sync.py" 2>/dev/null | head -1)
    fi

    if [[ -z "$sync_script" || ! -f "$sync_script" ]]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (no sync script)"
        csv_row "$model_name" py "$category" SKIP
        BENCH_SKIP=$((BENCH_SKIP + 1))
        return
    fi

    local log_file="${LOG_DIR}/${model_name}_py.log"
    local py_input_flag="--image"
    [[ "$input_flag" == "-v" ]] && py_input_flag="--video"
    local cmd=("${TIMEOUT_CMD[@]}" "${PYTHON}" "${sync_script}" --model "${model_file}" "${py_input_flag}" "${input_path}" --no-display)
    BENCH_TOTAL=$((BENCH_TOTAL + 1))

    # Warmup
    if (( WARMUP > 0 )); then
        echo -e "  ${DIM}[WarmUp]${NC} ${model_name} (py, ${WARMUP} runs)..."
        "${cmd[@]}" --loop "${WARMUP}" > /dev/null 2>&1 || true
    fi

    # Actual benchmark — one process, --loop N (same as the C++ -l N)
    echo -e "  ${BLUE}[Bench]${NC} ${model_name} (py, ${LOOPS} loops)..."
    "${cmd[@]}" --loop "${LOOPS}" > "${log_file}" 2>&1
    local rc=$?
    if (( rc == 0 )); then
        record_result "$model_name" py "$category" "$log_file"
    else
        record_failure "$model_name" py "$category" "$rc" "$log_file"
    fi
}

# ============================================================================
# Discover models
# ============================================================================
BENCH_LANGS=()
[[ "$LANG_MODE" == "cpp" || "$LANG_MODE" == "both" ]] && BENCH_LANGS+=(cpp)
[[ "$LANG_MODE" == "py"  || "$LANG_MODE" == "both" ]] && BENCH_LANGS+=(py)

declare -A CAT_MODELS=()  # category -> newline-separated variants
filter_lc="${FILTER,,}"
for name in "${!MODEL_FILE[@]}"; do
    if [[ -n "$FILTER" ]]; then
        name_lc="${name,,}"
        family_lc="${MODEL_FAMILY[$name],,}"
        [[ "$name_lc" != *"$filter_lc"* && "$family_lc" != *"$filter_lc"* ]] && continue
    fi
    category="${MODEL_CATEGORY[$name]}"
    [[ -n "$CATEGORY_FILTER" && "$category" != "$CATEGORY_FILTER" ]] && continue
    CAT_MODELS["$category"]+="${name}"$'\n'
done

# Tasks in CATEGORY_ORDER first, then any other task alphabetically, so a task
# missing from CATEGORY_ORDER is still benchmarked rather than dropped.
ordered_categories=()
for category in "${CATEGORY_ORDER[@]}"; do
    [[ -n "${CAT_MODELS[$category]+x}" ]] && ordered_categories+=("$category")
done
while IFS= read -r category; do
    [[ -z "$category" || " ${CATEGORY_ORDER[*]} " == *" $category "* ]] && continue
    ordered_categories+=("$category")
done < <(printf '%s\n' "${!CAT_MODELS[@]}" | sort)

declare -a BENCH_MODELS=()  # "variant|category"
for category in "${ordered_categories[@]}"; do
    while IFS= read -r name; do
        [[ -n "$name" ]] && BENCH_MODELS+=("${name}|${category}")
    done < <(printf '%s' "${CAT_MODELS[$category]}" | sort)
done

# ============================================================================
# Header
# ============================================================================
{
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║${NC}   ${CYAN}DX-APP Benchmark Suite${NC}                         ${BOLD}║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  Language  : ${GREEN}${LANG_MODE}${NC}$(printf '%*s' $((35 - ${#LANG_MODE})) '')${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Loops     : ${GREEN}${LOOPS}${NC}$(printf '%*s' $((35 - ${#LOOPS})) '')${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Warmup    : ${GREEN}${WARMUP}${NC}$(printf '%*s' $((35 - ${#WARMUP})) '')${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Input     : ${GREEN}$(if $USE_VIDEO; then echo "video"; else echo "image"; fi)${NC}$(printf '%*s' $((30)) '')${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Models    : ${GREEN}${#BENCH_MODELS[@]}${NC}$(printf '%*s' $((35 - ${#BENCH_MODELS[@]})) '')${BOLD}║${NC}"
    [[ -n "$FILTER" ]] && \
    echo -e "${BOLD}║${NC}  Filter    : ${YELLOW}${FILTER}${NC}$(printf '%*s' $((35 - ${#FILTER})) '')${BOLD}║${NC}"
    [[ -n "$CATEGORY_FILTER" ]] && \
    echo -e "${BOLD}║${NC}  Task      : ${YELLOW}${CATEGORY_FILTER}${NC}$(printf '%*s' $((35 - ${#CATEGORY_FILTER})) '')${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Output    : ${DIM}${LOG_DIR}${NC}$(printf '%*s' $((35 - ${#LOG_DIR})) '')${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Started   : ${DIM}$(date)${NC}    ${BOLD}║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
    echo -e "  C++ bin   : ${DIM}${BUILD_DIR}${NC}"
    echo -e "  Python    : ${DIM}${PYTHON}${NC}"
    echo ""
} | tee "${SUMMARY_FILE}"

# ============================================================================
# Run benchmarks grouped by category
# ============================================================================
current_category=""

for entry in "${BENCH_MODELS[@]}"; do
    IFS='|' read -r model_name category <<< "$entry"
    model_file="${MODEL_FILE[$model_name]}"
    family="${MODEL_FAMILY[$model_name]}"

    # Category header
    if [[ "$category" != "$current_category" ]]; then
        current_category="$category"
        display_name="${CATEGORY_DISPLAY[$category]:-$category}"
        echo ""
        echo -e "${BLUE}━━━ ${display_name} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "${SUMMARY_FILE}"
    fi

    # Determine input (an image input may be a directory, e.g. face_pair)
    if $USE_VIDEO && ! is_image_only_category "$category"; then
        input_flag="-v"
        input_path="${CATEGORY_VIDEO[$category]}"
    else
        input_flag="-i"
        input_path="${MODEL_IMAGE_OVERRIDE[$model_name]:-${CATEGORY_IMAGE[$category]:-sample/img/sample_street.jpg}}"
    fi
    if [[ -z "$input_path" || ! -e "$input_path" ]]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (input not found: ${input_path:-<none for ${category}>})"
        for lang in "${BENCH_LANGS[@]}"; do
            csv_row "$model_name" "$lang" "$category" NO_INPUT
        done
        BENCH_SKIP=$((BENCH_SKIP + 1))
        continue
    fi

    # Verify model file
    if [[ -z "$model_file" || ! -f "$model_file" ]]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (model file not found: ${model_file})"
        for lang in "${BENCH_LANGS[@]}"; do
            csv_row "$model_name" "$lang" "$category" NO_MODEL
        done
        BENCH_SKIP=$((BENCH_SKIP + 1))
        continue
    fi

    # C++ benchmark
    if [[ "$LANG_MODE" == "cpp" || "$LANG_MODE" == "both" ]]; then
        run_bench_cpp "$model_name" "$model_file" "$category" "$input_flag" "$input_path"
    fi

    # Python benchmark
    if [[ "$LANG_MODE" == "py" || "$LANG_MODE" == "both" ]]; then
        run_bench_py "$model_name" "$model_file" "$category" "$family" "$input_flag" "$input_path"
    fi
done

# ============================================================================
# Results Table — Generate pretty table using Python
# ============================================================================
generate_summary_table() {
    "$PYTHON" << 'PYEOF'
import csv, sys, os

csv_path = os.environ.get("CSV_FILE", "")
compare_csv = os.environ.get("COMPARE_CSV", "")
csv_only = os.environ.get("CSV_ONLY", "false") == "true"

if not csv_path or not os.path.exists(csv_path):
    print("No results CSV found")
    sys.exit(0)

# Load current results
rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get("status") == "OK":
            rows.append(r)

if not rows:
    print("\nNo successful benchmark results.")
    sys.exit(0)

# Load comparison data (if any)
prev = {}
if compare_csv and os.path.exists(compare_csv):
    with open(compare_csv) as f:
        for r in csv.DictReader(f):
            if r.get("status") == "OK":
                key = f"{r['model']}_{r['lang']}"
                prev[key] = r

# Format table (variant names run past 22 chars, so size the text columns)
mw = max([22] + [len(r["model"]) for r in rows])
cw = max([22] + [len(r.get("category", "")) for r in rows])
width = mw + cw + 63
lines = []
lines.append("")
lines.append("=" * width)
lines.append(f"{'BENCHMARK RESULTS':^{width}}")
lines.append("=" * width)

header = f"{'Model':<{mw}} {'Lang':<5} {'Category':<{cw}} {'Infer(ms)':>9} {'InferFPS':>9} {'Pre(ms)':>8} {'Post(ms)':>9} {'Overall':>8} {'Frames':>7}"
if prev:
    header += f" {'Δ Infer':>8}"
lines.append(header)
lines.append("-" * width + ("-" * 9 if prev else ""))

current_cat = ""
for r in rows:
    cat = r.get("category", "")
    if cat != current_cat:
        if current_cat:
            lines.append("")
        current_cat = cat

    infer_ms = r.get("infer_ms", "")
    infer_fps = r.get("infer_fps", "")
    pre_ms = r.get("pre_ms", "")
    post_ms = r.get("post_ms", "")
    overall = r.get("overall_fps", "")
    frames = r.get("frames", "")

    line = f"{r['model']:<{mw}} {r['lang']:<5} {cat:<{cw}} "
    line += f"{infer_ms:>9} {infer_fps:>9} {pre_ms:>8} {post_ms:>9} {overall:>8} {frames:>7}"

    if prev:
        key = f"{r['model']}_{r['lang']}"
        if key in prev and infer_ms and prev[key].get("infer_ms"):
            try:
                delta = float(infer_ms) - float(prev[key]["infer_ms"])
                sign = "+" if delta > 0 else ""
                # Positive delta = slower (bad), negative = faster (good)
                color = "\033[0;31m" if delta > 0 else "\033[0;32m"
                line += f" {color}{sign}{delta:.2f}\033[0m"
            except:
                line += f" {'':>8}"
        else:
            line += f" {'NEW':>8}"

    lines.append(line)

lines.append("-" * width + ("-" * 9 if prev else ""))

# Aggregated stats
cats = {}
for r in rows:
    cat = r.get("category", "unknown")
    lang = r.get("lang", "?")
    key = f"{cat} ({lang})"
    if key not in cats:
        cats[key] = {"count": 0, "sum_infer": 0, "sum_overall": 0}
    cats[key]["count"] += 1
    try:
        if r.get("infer_ms"):
            cats[key]["sum_infer"] += float(r["infer_ms"])
        if r.get("overall_fps"):
            cats[key]["sum_overall"] += float(r["overall_fps"])
    except:
        pass

lines.append("")
lines.append(f"{'CATEGORY AVERAGES':^{width}}")
lines.append("-" * 75)
lines.append(f"{'Category':<40} {'Count':>6} {'Avg Infer(ms)':>14} {'Avg Overall FPS':>16}")
lines.append("-" * 75)
for key in sorted(cats.keys()):
    c = cats[key]
    avg_infer = c["sum_infer"] / c["count"] if c["count"] > 0 else 0
    avg_overall = c["sum_overall"] / c["count"] if c["count"] > 0 else 0
    lines.append(f"{key:<40} {c['count']:>6} {avg_infer:>14.2f} {avg_overall:>16.1f}")
lines.append("-" * 75)

# Grand totals
total_ok = len(rows)
lines.append("")
lines.append(f"Total benchmarked: {total_ok}")
if prev:
    lines.append(f"Comparison base: {compare_csv}")
lines.append("=" * width)

for l in lines:
    print(l)

return_text = "\n".join(lines)
PYEOF
}

echo "" | tee -a "${SUMMARY_FILE}"

if [[ "$CSV_ONLY" != true ]]; then
    echo -e "${BOLD}Generating benchmark results table...${NC}"
    CSV_FILE="$CSV_FILE" COMPARE_CSV="$COMPARE_CSV" CSV_ONLY="$CSV_ONLY" \
        generate_summary_table | tee -a "${SUMMARY_FILE}"
fi

# ============================================================================
# Final Summary
# ============================================================================
{
    echo ""
    echo "========================================"
    echo "  Benchmark Completed: $(date)"
    echo "========================================"
    echo -e "  Total: ${BENCH_TOTAL}  ${GREEN}OK: ${BENCH_OK}${NC}  ${RED}FAIL: ${BENCH_FAIL}${NC}  ${YELLOW}SKIP: ${BENCH_SKIP}${NC}"
    echo ""
    echo "  Results CSV : ${CSV_FILE}"
    echo "  Summary     : ${SUMMARY_FILE}"
    if [[ -s "$ERROR_LOG" ]]; then
        echo -e "  ${RED}Errors      : ${ERROR_LOG}${NC}"
    fi
    echo "========================================"
} | tee -a "${SUMMARY_FILE}"

# Remove empty error log
[[ ! -s "$ERROR_LOG" ]] && rm -f "$ERROR_LOG"

exit 0
