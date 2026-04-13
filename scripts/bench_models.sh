#!/bin/bash
# =============================================================================
# bench_models.sh - DX-APP Model Benchmark Script
# =============================================================================
# Automatically measures per-model inference speed and reports results as
# CSV + summary table. Targets models registered in config/test_models.conf,
# running C++/Python sync runners in --no-display mode and parsing the
# PERFORMANCE SUMMARY output.
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
#     ├── <model_name>.log      Per-model execution log
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
    CANDIDATE_DIRS=(
        "src/cpp_example/build"
        "bin"
        "build_x86_64/release/bin"
        "build_x86_64/bin"
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
    if [ -z "${BUILD_DIR}" ]; then
        BUILD_DIR="src/cpp_example/build"
    fi
fi
PY_BASE="src/python_example"
CONFIG_FILE="${PROJECT_ROOT}/config/test_models.conf"

# Colors
RED='\033[0;31m';    GREEN='\033[0;32m';  YELLOW='\033[1;33m'
BLUE='\033[0;34m';   CYAN='\033[0;36m';   MAGENTA='\033[0;35m'
BOLD='\033[1m';      DIM='\033[2m';       NC='\033[0m'

# ============================================================================
# Category → Default Input Mapping (same as run_examples.sh)
# ============================================================================
declare -A CATEGORY_IMAGE CATEGORY_VIDEO CATEGORY_DISPLAY
CATEGORY_IMAGE=(
    [object_detection]="sample/img/sample_street.jpg"
    [face_detection]="sample/img/sample_face.jpg"
    [pose_estimation]="sample/img/sample_people.jpg"
    [obb_detection]="sample/img/sample_parking.jpg"
    [classification]="sample/img/sample_dog.jpg"
    [instance_segmentation]="sample/img/sample_street.jpg"
    [semantic_segmentation]="sample/img/sample_people.jpg"
    [depth_estimation]="sample/img/sample_horse.jpg"
    [image_denoising]="sample/img/sample_dark_room.jpg"
    [super_resolution]="sample/img/sample_dark_room.jpg"
    [image_enhancement]="sample/img/sample_dark_room.jpg"
    [embedding]="sample/img/sample_face_a1.jpg"
    [ppu]="sample/img/sample_street.jpg"
    [hand_landmark]="sample/img/sample_people.jpg"
    [attribute_recognition]="sample/img/sample_person.jpg"
    [reid]="sample/img/sample_person.jpg"
)
CATEGORY_VIDEO=(
    [object_detection]="assets/videos/dance-group.mov"
    [face_detection]="assets/videos/dance-solo.mov"
    [pose_estimation]="assets/videos/dance-solo.mov"
    [obb_detection]="assets/videos/dance-group.mov"
    [classification]="assets/videos/dance-group.mov"
    [instance_segmentation]="assets/videos/dance-group.mov"
    [semantic_segmentation]="assets/videos/blackbox-city-road.mp4"
    [depth_estimation]="assets/videos/dance-group.mov"
    [image_denoising]="assets/videos/dance-group.mov"
    [super_resolution]="assets/videos/dance-group.mov"
    [image_enhancement]="assets/videos/dance-group.mov"
    [embedding]="assets/videos/dance-group.mov"
    [ppu]="assets/videos/dance-group.mov"
    [hand_landmark]="assets/videos/hand.mp4"
    [attribute_recognition]="assets/videos/dance-group.mov"
    [reid]="assets/videos/dance-group.mov"
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
    [ppu]="PPU (Post-Processing Unit)"
    [hand_landmark]="Hand Landmark"
    [attribute_recognition]="Attribute Recognition"
    [reid]="Person Re-Identification"
)

CATEGORY_ORDER=(
    object_detection face_detection pose_estimation obb_detection classification
    instance_segmentation semantic_segmentation depth_estimation
    image_denoising super_resolution image_enhancement
    embedding ppu
    hand_landmark
    attribute_recognition reid
)

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
        --category)     CATEGORY_FILTER="$2"; shift 2 ;;
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
  --filter, -f <STR>      Only benchmark models matching STR
  --category <CAT>        Only benchmark models in specified category
  --video                 Use video input instead of image (longer test)
  --output-dir <DIR>      Custom output directory (default: logs/bench_TIMESTAMP)
  --csv-only              Output CSV only (skip summary table)
  --compare <CSV>         Compare with previous results CSV
  -h, --help              Show this help

Output:
  logs/bench_YYYYMMDD_HHMMSS/
    results.csv           Full results in CSV format
    summary.txt           Human-readable summary table
    <model>.log           Per-model execution log
    errors.log            Errors (if any)

Performance Summary Columns:
  Model          : Model name (e.g., yolov8)
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

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}Error:${NC} python3 is required for result parsing"
    exit 1
fi

# ============================================================================
# Load Model Registry
# ============================================================================
declare -A MODEL_FILE MODEL_CATEGORY

while IFS=$'\t' read -r name category model_file; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    MODEL_FILE["$name"]="$model_file"
    MODEL_CATEGORY["$name"]="$category"
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
    python3 -c "
import re, sys

text = open('$log_file', encoding='utf-8', errors='replace').read()

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
" 2>/dev/null
}

# ============================================================================
# run_bench_cpp — Benchmark a single C++ model
# ============================================================================
run_bench_cpp() {
    local model_name="$1"
    local model_file="$2"
    local category="$3"
    local input_arg="$4"
    local sync_exe="${BUILD_DIR}/${model_name}_sync"

    if [ ! -f "${sync_exe}" ]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (no sync binary)"
        echo "${model_name},cpp,${category},,,,,,0,SKIP" >> "$CSV_FILE"
        BENCH_SKIP=$((BENCH_SKIP + 1))
        return
    fi

    local log_file="${LOG_DIR}/${model_name}_cpp.log"
    BENCH_TOTAL=$((BENCH_TOTAL + 1))

    # Warmup
    if (( WARMUP > 0 )); then
        echo -e "  ${DIM}[WarmUp]${NC} ${model_name} (${WARMUP} runs)..."
        ${sync_exe} -m "${model_file}" ${input_arg} --no-display -l ${WARMUP} > /dev/null 2>&1 || true
    fi

    # Actual benchmark
    echo -e "  ${BLUE}[Bench]${NC} ${model_name} (cpp, ${LOOPS} loops)..."
    if ${sync_exe} -m "${model_file}" ${input_arg} --no-display -l ${LOOPS} > "${log_file}" 2>&1; then
        local result
        result=$(parse_performance_summary "$log_file")
        if [[ -n "$result" && "$result" != "|||||||" ]]; then
            IFS='|' read -r infer_ms infer_fps pre_ms post_ms overall_fps frames <<< "$result"
            echo "${model_name},cpp,${category},${infer_ms},${infer_fps},${pre_ms},${post_ms},${overall_fps},${frames},OK" >> "$CSV_FILE"
            echo -e "    ${GREEN}✓${NC} Infer: ${CYAN}${infer_ms}${NC} ms (${CYAN}${infer_fps}${NC} FPS)  Overall: ${CYAN}${overall_fps}${NC} FPS  [${frames} frames]"
            BENCH_OK=$((BENCH_OK + 1))
        else
            echo "${model_name},cpp,${category},,,,,,0,PARSE_ERROR" >> "$CSV_FILE"
            echo -e "    ${YELLOW}⚠${NC} Failed to parse performance data"
            echo "[PARSE_ERROR] ${model_name} cpp — no PERFORMANCE SUMMARY found" >> "$ERROR_LOG"
            BENCH_FAIL=$((BENCH_FAIL + 1))
        fi
    else
        local exit_code=$?
        echo "${model_name},cpp,${category},,,,,,0,EXEC_ERROR" >> "$CSV_FILE"
        echo -e "    ${RED}✗${NC} Execution failed (exit: ${exit_code})"
        echo "[EXEC_ERROR] ${model_name} cpp — exit code ${exit_code}" >> "$ERROR_LOG"
        BENCH_FAIL=$((BENCH_FAIL + 1))
    fi
}

# ============================================================================
# run_bench_py — Benchmark a single Python model
# ============================================================================
run_bench_py() {
    local model_name="$1"
    local model_file="$2"
    local category="$3"
    local input_arg="$4"

    # Find Python model directory
    local model_dir=""
    for cat_dir in "${PY_BASE}"/*/; do
        local cat=$(basename "$cat_dir")
        [[ "$cat" == "common" || "$cat" == "__pycache__" || "$cat" == "utils" ]] && continue
        if [[ -d "${cat_dir}${model_name}" ]]; then
            model_dir="${cat_dir}${model_name}"
            break
        fi
    done

    if [[ -z "$model_dir" || ! -d "$model_dir" ]]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (no Python directory)"
        echo "${model_name},py,${category},,,,,,0,SKIP" >> "$CSV_FILE"
        BENCH_SKIP=$((BENCH_SKIP + 1))
        return
    fi

    # Find sync script
    local sync_script=""
    sync_script=$(find "$model_dir" -maxdepth 1 -name "${model_name}_sync.py" -type f 2>/dev/null | head -1)
    if [[ -z "$sync_script" ]]; then
        # Fallback: find any *_sync.py in the directory
        sync_script=$(find "$model_dir" -maxdepth 1 -name "*_sync.py" -type f 2>/dev/null | head -1)
    fi

    if [[ -z "$sync_script" ]]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (no sync script)"
        echo "${model_name},py,${category},,,,,,0,SKIP" >> "$CSV_FILE"
        BENCH_SKIP=$((BENCH_SKIP + 1))
        return
    fi

    local log_file="${LOG_DIR}/${model_name}_py.log"
    BENCH_TOTAL=$((BENCH_TOTAL + 1))

    # Convert Python input args (-i → --image, -v → --video)
    local py_input_arg
    py_input_arg=$(echo "$input_arg" | sed 's/^-i /--image /' | sed 's/^-v /--video /')

    # Warmup
    if (( WARMUP > 0 )); then
        echo -e "  ${DIM}[WarmUp]${NC} ${model_name} (py, ${WARMUP} runs)..."
        python3 "${sync_script}" --model "${model_file}" ${py_input_arg} --no-display > /dev/null 2>&1 || true
    fi

    # Actual benchmark — Python sync runner runs once per image, so we loop
    echo -e "  ${BLUE}[Bench]${NC} ${model_name} (py, ${LOOPS} loops)..."

    # Run multiple times, use last result (video mode runs once)
    local run_count=${LOOPS}
    if [[ "$USE_VIDEO" == true ]]; then
        run_count=1
    fi

    local final_log="${log_file}"
    local i=0
    for (( i=1; i<=run_count; i++ )); do
        local iter_log="${LOG_DIR}/${model_name}_py_iter${i}.log"
        python3 "${sync_script}" --model "${model_file}" ${py_input_arg} --no-display > "${iter_log}" 2>&1 || true
        final_log="${iter_log}"
    done

    # Merge results into final log
    if (( run_count > 1 )); then
        cat "${LOG_DIR}/${model_name}_py_iter"*.log > "${log_file}" 2>/dev/null
        rm -f "${LOG_DIR}/${model_name}_py_iter"*.log
    else
        mv "${final_log}" "${log_file}" 2>/dev/null || true
    fi

    # Parse
    local result
    result=$(parse_performance_summary "$log_file")
    if [[ -n "$result" && "$result" != "|||||||" ]]; then
        IFS='|' read -r infer_ms infer_fps pre_ms post_ms overall_fps frames <<< "$result"
        echo "${model_name},py,${category},${infer_ms},${infer_fps},${pre_ms},${post_ms},${overall_fps},${frames},OK" >> "$CSV_FILE"
        echo -e "    ${GREEN}✓${NC} Infer: ${CYAN}${infer_ms}${NC} ms (${CYAN}${infer_fps}${NC} FPS)  Overall: ${CYAN}${overall_fps}${NC} FPS  [${frames} frames]"
        BENCH_OK=$((BENCH_OK + 1))
    else
        echo "${model_name},py,${category},,,,,,0,PARSE_ERROR" >> "$CSV_FILE"
        echo -e "    ${YELLOW}⚠${NC} Failed to parse performance data"
        echo "[PARSE_ERROR] ${model_name} py — no PERFORMANCE SUMMARY found" >> "$ERROR_LOG"
        BENCH_FAIL=$((BENCH_FAIL + 1))
    fi
}

# ============================================================================
# Discover models
# ============================================================================
declare -a BENCH_MODELS=()  # "model_name|category"

for name in "${!MODEL_FILE[@]}"; do
    [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
    [[ -n "$CATEGORY_FILTER" && "${MODEL_CATEGORY[$name]}" != "$CATEGORY_FILTER" ]] && continue
    BENCH_MODELS+=("$name|${MODEL_CATEGORY[$name]}")
done

# Sort by category order
sorted_models=()
for category in "${CATEGORY_ORDER[@]}"; do
    for entry in "${BENCH_MODELS[@]}"; do
        IFS='|' read -r m_name m_cat <<< "$entry"
        [[ "$m_cat" == "$category" ]] && sorted_models+=("$entry")
    done
done
BENCH_MODELS=("${sorted_models[@]}")

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
    echo -e "${BOLD}║${NC}  Output    : ${DIM}${LOG_DIR}${NC}$(printf '%*s' $((35 - ${#LOG_DIR})) '')${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Started   : ${DIM}$(date)${NC}    ${BOLD}║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
} | tee "${SUMMARY_FILE}"

# ============================================================================
# Run benchmarks grouped by category
# ============================================================================
current_category=""

for entry in "${BENCH_MODELS[@]}"; do
    IFS='|' read -r model_name category <<< "$entry"
    model_file="${MODEL_FILE[$model_name]}"

    # Category header
    if [[ "$category" != "$current_category" ]]; then
        current_category="$category"
        display_name="${CATEGORY_DISPLAY[$category]:-$category}"
        echo ""
        echo -e "${BLUE}━━━ ${display_name} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "${SUMMARY_FILE}"
    fi

    # Determine input args
    input_arg=""
    if $USE_VIDEO; then
        video="${CATEGORY_VIDEO[$category]}"
        if [[ -z "$video" || ! -f "$video" ]]; then
            echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (video file not found: ${video})"
            BENCH_SKIP=$((BENCH_SKIP + 1))
            continue
        fi
        input_arg="-v ${video}"
    else
        image="${CATEGORY_IMAGE[$category]}"
        if [[ -z "$image" || ! -f "$image" ]]; then
            echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (image file not found: ${image})"
            BENCH_SKIP=$((BENCH_SKIP + 1))
            continue
        fi
        input_arg="-i ${image}"
    fi

    # Verify model file
    if [[ -z "$model_file" || ! -f "$model_file" ]]; then
        echo -e "  ${YELLOW}[SKIP]${NC} ${model_name} (model file not found: ${model_file})"
        echo "${model_name},*,${category},,,,,,0,NO_MODEL" >> "$CSV_FILE"
        BENCH_SKIP=$((BENCH_SKIP + 1))
        continue
    fi

    # C++ benchmark
    if [[ "$LANG_MODE" == "cpp" || "$LANG_MODE" == "both" ]]; then
        run_bench_cpp "$model_name" "$model_file" "$category" "$input_arg"
    fi

    # Python benchmark
    if [[ "$LANG_MODE" == "py" || "$LANG_MODE" == "both" ]]; then
        run_bench_py "$model_name" "$model_file" "$category" "$input_arg"
    fi
done

# ============================================================================
# Results Table — Generate pretty table using Python
# ============================================================================
generate_summary_table() {
    python3 << 'PYEOF'
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

# Format table
lines = []
lines.append("")
lines.append("=" * 100)
lines.append(f"{'BENCHMARK RESULTS':^100}")
lines.append("=" * 100)

header = f"{'Model':<22} {'Lang':<5} {'Category':<22} {'Infer(ms)':>9} {'InferFPS':>9} {'Pre(ms)':>8} {'Post(ms)':>9} {'Overall':>8} {'Frames':>7}"
if prev:
    header += f" {'Δ Infer':>8}"
lines.append(header)
lines.append("-" * 100 + ("-" * 9 if prev else ""))

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

    line = f"{r['model']:<22} {r['lang']:<5} {cat:<22} "
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

lines.append("-" * 100 + ("-" * 9 if prev else ""))

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
lines.append(f"{'CATEGORY AVERAGES':^100}")
lines.append("-" * 70)
lines.append(f"{'Category':<35} {'Count':>6} {'Avg Infer(ms)':>14} {'Avg Overall FPS':>16}")
lines.append("-" * 70)
for key in sorted(cats.keys()):
    c = cats[key]
    avg_infer = c["sum_infer"] / c["count"] if c["count"] > 0 else 0
    avg_overall = c["sum_overall"] / c["count"] if c["count"] > 0 else 0
    lines.append(f"{key:<35} {c['count']:>6} {avg_infer:>14.2f} {avg_overall:>16.1f}")
lines.append("-" * 70)

# Grand totals
total_ok = len(rows)
lines.append("")
lines.append(f"Total benchmarked: {total_ok}")
if prev:
    lines.append(f"Comparison base: {compare_csv}")
lines.append("=" * 100)

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
