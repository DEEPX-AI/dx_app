#!/bin/bash
# ============================================================================
# validate_models.sh (C++/Python) - Registry-driven
# Runs add_model.sh --verify for all models based on model_registry.json
#
# Usage:
#   ./validate_models.sh                        # C++ + Python (all models)
#   ./validate_models.sh --lang cpp             # C++ only
#   ./validate_models.sh --lang py              # Python only
#   ./validate_models.sh classification         # Filter by task
#   ./validate_models.sh --list                 # Print commands only
#   ./validate_models.sh --list --lang py       # Print Python commands only
#   ./validate_models.sh --clean                # Remove all generated packages
#   ./validate_models.sh --clean --lang py      # Remove Python packages only
#   ./validate_models.sh --no-video             # Image-only (no video input)
#   ./validate_models.sh --skip-verify          # Code generation only (no NPU inference)
#   ./validate_models.sh --numerical            # Run + numerical verification
# ============================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DX_APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_SRC_DIR="$DX_APP_ROOT/src/cpp_example"
PY_SRC_DIR="$DX_APP_ROOT/src/python_example"
REGISTRY="$DX_APP_ROOT/config/model_registry.json"
ASSET_DIR="$DX_APP_ROOT/assets/models"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
SYM_WARN="⚠"

# ============================================================================
# Parse arguments
# ============================================================================
FILTER="all"
LANG_MODE="both"
LIST_ONLY=false
CLEAN_MODE=false
NO_VIDEO=false
SKIP_VERIFY=false
NUMERICAL=false
START_FROM=""

while [ $# -gt 0 ]; do
    case "$1" in
        --lang)         LANG_MODE="$2"; shift 2 ;;
        --start-from)   START_FROM="$2"; shift 2 ;;
        --list)         LIST_ONLY=true; shift ;;
        --clean)        CLEAN_MODE=true; shift ;;
        --no-video)     NO_VIDEO=true; shift ;;
        --image-only)   NO_VIDEO=true; shift ;;
        --skip-verify)  SKIP_VERIFY=true; shift ;;
        --numerical)    NUMERICAL=true; shift ;;
        --help|-h)
            cat << 'HELPEOF'
validate_models.sh (C++/Python) - Registry-driven model validator

Usage:
  ./scripts/validate_models.sh                        # C++ + Python (all models)
  ./scripts/validate_models.sh --lang cpp             # C++ only
  ./scripts/validate_models.sh --lang py              # Python only
  ./scripts/validate_models.sh classification         # Filter by task
  ./scripts/validate_models.sh --list                 # Print commands only
  ./scripts/validate_models.sh --list --lang py       # Print Python commands only
  ./scripts/validate_models.sh --clean                # Remove all generated packages
  ./scripts/validate_models.sh --clean --lang py      # Remove Python packages only
  ./scripts/validate_models.sh --no-video             # Image-only (no video input)
  ./scripts/validate_models.sh --skip-verify          # Code generation only (no NPU inference)
  ./scripts/validate_models.sh --numerical            # Run + numerical verification
  ./scripts/validate_models.sh --start-from MODEL     # Resume from a specific model
  -h, --help                                          # Show this help

Valid task filters:
  all classification detection face_detection instance_segmentation
  semantic_segmentation pose depth image_denoising super_resolution
  image_enhancement obb embedding ppu hand_landmark
HELPEOF
            exit 0
            ;;
        --*)            echo "Unknown option: $1"; exit 1 ;;
        *)              FILTER="$1"; shift ;;
    esac
done

case "$LANG_MODE" in
    cpp)  LANGS=("cpp") ;;
    py)   LANGS=("py") ;;
    both) LANGS=("cpp" "py") ;;
    *)    echo "Invalid --lang: $LANG_MODE (cpp|py|both)"; exit 1 ;;
esac

# ============================================================================
# Check dependencies
# ============================================================================
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}[ERROR] python3 not found${NC}"; exit 1
fi
if [ ! -f "$REGISTRY" ]; then
    echo -e "${RED}[ERROR] Registry not found: $REGISTRY${NC}"; exit 1
fi

TOTAL=0; PASS=0; FAIL=0; SKIP=0
CONFIG_WARN=0
NUM_PASS=0; NUM_FAIL=0; NUM_WARN=0; NUM_SKIP=0
FAILED_MODELS=()
RMAP_FAILED_MODELS=()
NUM_FAILED_MODELS=()
CONFIG_WARNINGS=()

# For cleanup
TEST_PACKAGES=()
declare -A PKG_CATEGORIES

# Current language context
CURRENT_LANG=""
CURRENT_SRC_DIR=""

# Log directory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LANG_TAG=$(echo "$LANG_MODE" | tr '[:upper:]' '[:lower:]')
VERIFY_TAG=$( [ "$SKIP_VERIFY" = true ] && echo "_noverify" || echo "" )
if [ "$NUMERICAL" = true ]; then VERIFY_TAG="${VERIFY_TAG}_numerical"; fi
LOG_DIR="$DX_APP_ROOT/logs/validate_${LANG_TAG}${VERIFY_TAG}_${TIMESTAMP}"
SUMMARY_LOG="$LOG_DIR/summary.log"
LIVE_LOG="$LOG_DIR/live_progress.log"
if [ "$LIST_ONLY" = false ] && [ "$CLEAN_MODE" = false ]; then
    mkdir -p "$LOG_DIR"
    : > "$LIVE_LOG"
fi

log_live() {
    local msg="$1"
    if [ "$LIST_ONLY" = false ] && [ "$CLEAN_MODE" = false ] && [ -n "$LIVE_LOG" ]; then
        printf '%b\n' "$msg" >> "$LIVE_LOG"
    fi
}

# ============================================================================
# task_to_category - add_model_task → src directory name
# ============================================================================
task_to_category() {
    case "$1" in
        classification)          echo "classification" ;;
        detection)               echo "object_detection" ;;
        face_detection)          echo "face_detection" ;;
        pose)                    echo "pose_estimation" ;;
        obb)                     echo "obb_detection" ;;
        semantic_segmentation)   echo "semantic_segmentation" ;;
        instance_segmentation)   echo "instance_segmentation" ;;
        depth_estimation)        echo "depth_estimation" ;;
        image_denoising)         echo "image_denoising" ;;
        super_resolution)        echo "super_resolution" ;;
        image_enhancement)       echo "image_enhancement" ;;
        embedding)               echo "embedding" ;;
        ppu)                     echo "ppu" ;;
        hand_landmark)           echo "hand_landmark" ;;
        *)                       echo "$1" ;;
    esac
}

# ============================================================================
# filter_to_task - CLI filter name → registry add_model_task
# ============================================================================
filter_to_task() {
    case "$1" in
        classification)         echo "classification" ;;
        detection|object_detection) echo "object_detection" ;;
        face_detection)         echo "face_detection" ;;
        pose|pose_estimation)   echo "pose_estimation" ;;
        obb|obb_detection)      echo "obb_detection" ;;
        semantic_segmentation)  echo "semantic_segmentation" ;;
        instance_segmentation)  echo "instance_segmentation" ;;
        depth|depth_estimation) echo "depth_estimation" ;;
        image_denoising)        echo "image_denoising" ;;
        super_resolution)       echo "super_resolution" ;;
        image_enhancement)      echo "image_enhancement" ;;
        embedding)              echo "embedding" ;;
        ppu)                    echo "ppu" ;;
        hand_landmark|hand)     echo "hand_landmark" ;;
        all)                    echo "" ;;
        *)  echo -e "${RED}Unknown filter: $1${NC}" >&2
            echo "Valid: all|classification|detection|face_detection|instance_segmentation|semantic_segmentation|pose|depth|image_denoising|super_resolution|image_enhancement|obb|embedding|ppu|hand_landmark" >&2
            exit 1 ;;
    esac
}

# ============================================================================
# validate_config - config.json validation
# ============================================================================
KNOWN_CONFIG_KEYS="score_threshold nms_threshold obj_threshold num_classes has_background reg_max top_k per_class_nms num_keypoints stride num_protos strides"

validate_config() {
    local name="$1" category="$2"
    local cpp_config="$CPP_SRC_DIR/$category/$name/config.json"
    local py_config="$PY_SRC_DIR/$category/$name/config.json"
    local issues=()

    local config_path
    if [[ "$CURRENT_LANG" == "cpp" ]]; then
        config_path="$cpp_config"
    else
        config_path="$py_config"
    fi

    if [[ ! -f "$config_path" ]]; then
        issues+=("config.json not found")
    else
        if ! python3 -c "import json; json.load(open('$config_path'))" 2>/dev/null; then
            issues+=("Invalid JSON format")
        else
            local result
            result=$(python3 -c "
import json, sys
known = set('$KNOWN_CONFIG_KEYS'.split())
cfg = json.load(open('$config_path'))
issues = []
for k, v in cfg.items():
    if k not in known:
        issues.append(f'Unknown key: {k}')
    if k in ('score_threshold', 'nms_threshold', 'obj_threshold'):
        if not isinstance(v, (int, float)):
            issues.append(f'{k}: not a number ({type(v).__name__})')
        elif v < 0.0 or v > 1.0:
            issues.append(f'{k}={v} out of range (0.0~1.0)')
    if k == 'num_classes':
        if not isinstance(v, int) or v < 1:
            issues.append(f'{k}={v} invalid (positive int required)')
    if k == 'reg_max':
        if not isinstance(v, int) or v < 1:
            issues.append(f'{k}={v} invalid (positive int required)')
    if k == 'has_background':
        if not isinstance(v, bool):
            issues.append(f'{k}: not bool ({type(v).__name__})')
for i in issues:
    print(i)
" 2>/dev/null)
            if [[ -n "$result" ]]; then
                while IFS= read -r line; do
                    issues+=("$line")
                done <<< "$result"
            fi
        fi
    fi

    # C++ / Python config mismatch check
    if [[ -f "$cpp_config" && -f "$py_config" ]]; then
        local cpp_hash py_hash
        cpp_hash=$(python3 -c "import json; print(json.dumps(json.load(open('$cpp_config')), sort_keys=True))" 2>/dev/null)
        py_hash=$(python3 -c "import json; print(json.dumps(json.load(open('$py_config')), sort_keys=True))" 2>/dev/null)
        if [[ -n "$cpp_hash" && -n "$py_hash" && "$cpp_hash" != "$py_hash" ]]; then
            issues+=("C++/Python config.json mismatch")
        fi
    fi

    if [[ ${#issues[@]} -gt 0 ]]; then
        CONFIG_WARN=$((CONFIG_WARN + 1))
        echo -e "  ${YELLOW}[CONFIG]${NC} $name warnings:"
        log_live "  [CONFIG] $name warnings:"
        for issue in "${issues[@]}"; do
            echo -e "    ${YELLOW}${SYM_WARN}${NC} $issue"
            log_live "    ${SYM_WARN} $issue"
            CONFIG_WARNINGS+=("$name ($CURRENT_LANG): $issue")
        done
    fi
}

# ============================================================================
# run_add_model - validate a single model
# ============================================================================
run_add_model() {
    local name="$1" task="$2" pp="$3" model_path="$4"
    TEST_PACKAGES+=("$name")

    local category
    category=$(task_to_category "$task")
    PKG_CATEGORIES["$name"]="$category"

    if [ "$CLEAN_MODE" = true ]; then return; fi

    TOTAL=$((TOTAL + 1))

    if [ "$LIST_ONLY" = true ]; then
        if [ "$SKIP_VERIFY" = true ]; then
            echo "./add_model.sh $name $task --lang $CURRENT_LANG --postprocessor $pp --model \"$model_path\""
        else
            echo "./add_model.sh $name $task --lang $CURRENT_LANG --postprocessor $pp --verify ${NO_VIDEO:+--no-video }--model \"$model_path\""
        fi
        return
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  [$CURRENT_LANG][$TOTAL] $name ($task / $pp)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log_live ""
    log_live "[$CURRENT_LANG][$TOTAL] $name ($task / $pp)"

    if [ ! -f "$model_path" ]; then
        echo -e "${YELLOW}[SKIP]${NC} Model file not found: $model_path"
        log_live "[SKIP] Model file not found: $model_path"
        SKIP=$((SKIP + 1))
        return
    fi

    local model_log="$LOG_DIR/${CURRENT_LANG}_${name}.log"
    local model_err_log="$LOG_DIR/${CURRENT_LANG}_${name}.error.log"

    # Reset NPU rmap before each model (only when verify is enabled)
    if [ "$SKIP_VERIFY" = false ] && command -v dxrt-cli &>/dev/null; then
        if [ "$LIST_ONLY" = false ] && [ "$CLEAN_MODE" = false ] && [ -n "$LIVE_LOG" ]; then
            dxrt-cli -r 1 2>/dev/null | tee -a "$LIVE_LOG" || true
        else
            dxrt-cli -r 1 2>/dev/null || true
        fi
        sleep 1
    fi

    cd "$CURRENT_SRC_DIR"
    local verify_flags=""
    if [ "$SKIP_VERIFY" = false ]; then
        verify_flags="--verify ${NO_VIDEO:+--no-video}"
    fi

    # Set DXAPP_VERIFY=1 for numerical verification (Python only)
    local _env_prefix=""
    if [ "$NUMERICAL" = true ] && [ "$CURRENT_LANG" = "py" ] && [ "$SKIP_VERIFY" = false ]; then
        local verify_dir="$LOG_DIR/verify"
        mkdir -p "$verify_dir"
        _env_prefix="DXAPP_VERIFY=1 DXAPP_VERIFY_DIR=$verify_dir"
    fi

    if env ${_env_prefix} "$SCRIPT_DIR/add_model.sh" "$name" "$task" --lang "$CURRENT_LANG" \
        --postprocessor "$pp" $verify_flags --model "$model_path" 2>&1 \
        | tee "$model_log"; then
        echo -e "${GREEN}[PASS]${NC} $name ($CURRENT_LANG)"
        log_live "[PASS] $name ($CURRENT_LANG)"
        echo "PASS" >> "$model_log"
        PASS=$((PASS + 1))

        # --- Numerical verification ---
        if [ "$NUMERICAL" = true ] && [ "$CURRENT_LANG" = "py" ] && [ "$SKIP_VERIFY" = false ]; then
            local verify_json="$verify_dir/$(python3 -c "import os; print(os.path.splitext(os.path.basename('$model_path'))[0])").json"
            if [ -f "$verify_json" ]; then
                local num_result num_rc
                set +e
                num_result=$(python3 "$SCRIPT_DIR/verify_inference_output.py" "$verify_json" \
                    --rules "$SCRIPT_DIR/inference_verify_rules.json" 2>&1)
                num_rc=$?
                set -e
                echo "$num_result"
                if [ $num_rc -eq 0 ]; then
                    echo -e "  ${GREEN}[NUMERICAL PASS]${NC} $name"
                    log_live "  [NUMERICAL PASS] $name"
                    NUM_PASS=$((NUM_PASS + 1))
                elif [ $num_rc -eq 2 ]; then
                    echo -e "  ${YELLOW}[NUMERICAL WARN]${NC} $name"
                    log_live "  [NUMERICAL WARN] $name"
                    NUM_WARN=$((NUM_WARN + 1))
                else
                    echo -e "  ${RED}[NUMERICAL FAIL]${NC} $name"
                    log_live "  [NUMERICAL FAIL] $name"
                    NUM_FAIL=$((NUM_FAIL + 1))
                    NUM_FAILED_MODELS+=("$name")
                fi
            else
                echo -e "  ${YELLOW}[NUMERICAL SKIP]${NC} No verify JSON: $verify_json"
                log_live "  [NUMERICAL SKIP] No verify JSON"
                NUM_SKIP=$((NUM_SKIP + 1))
            fi
        fi

        validate_config "$name" "$category"
    else
        echo -e "${RED}[FAIL]${NC} $name ($CURRENT_LANG)"
        log_live "[FAIL] $name ($CURRENT_LANG)"
        # Copy to error log as well
        cp "$model_log" "$model_err_log" 2>/dev/null || true
        echo "FAIL" >> "$model_log"
        FAIL=$((FAIL + 1))
        FAILED_MODELS+=("$name ($CURRENT_LANG)")
        # Detect NPU rmap corruption failure
        if grep -q "failed to write model rmap parameters" "$model_log" 2>/dev/null; then
            RMAP_FAILED_MODELS+=("$name")
            echo -e "  ${YELLOW}[RMAP]${NC} NPU rmap failure detected — model may be unsupported"
            log_live "  [RMAP] NPU rmap failure detected - model may be unsupported"
        fi
        # Extra NPU recovery after crash/fail: segfault leaves shared memory dirty
        if [ "$SKIP_VERIFY" = false ] && command -v dxrt-cli &>/dev/null; then
            echo -e "  ${YELLOW}[NPU]${NC} Extra reset after failure..."
            log_live "  [NPU] Extra reset after failure..."
            if [ "$LIST_ONLY" = false ] && [ "$CLEAN_MODE" = false ] && [ -n "$LIVE_LOG" ]; then
                dxrt-cli -r 1 2>/dev/null | tee -a "$LIVE_LOG" || true
            else
                dxrt-cli -r 1 2>/dev/null || true
            fi
            sleep 1
            if [ "$LIST_ONLY" = false ] && [ "$CLEAN_MODE" = false ] && [ -n "$LIVE_LOG" ]; then
                dxrt-cli -r 1 2>/dev/null | tee -a "$LIVE_LOG" || true
            else
                dxrt-cli -r 1 2>/dev/null || true
            fi
            sleep 1
        fi
        validate_config "$name" "$category"
        local fail_config="$CURRENT_SRC_DIR/$category/$name/config.json"
        if [[ -f "$fail_config" ]]; then
            echo -e "  ${YELLOW}[DIAG]${NC} config.json:"
            echo -e "    ${YELLOW}$(cat "$fail_config")${NC}"
            log_live "  [DIAG] config.json:"
            log_live "    $(cat "$fail_config")"
        fi
    fi
}

# ============================================================================
# run_from_registry - read model_registry.json and run validation
# ============================================================================
run_from_registry() {
    local filter_task="$1"  # empty string = all tasks

    # Count models for this filter
    local model_count
    model_count=$(python3 -c "
import json
with open('$REGISTRY') as f:
    models = json.load(f)
ft = '$filter_task'
if ft:
    models = [m for m in models if m['add_model_task'] == ft]
models = [m for m in models if m.get('supported', True)]
print(len(models))
")
    local filter_label="${filter_task:-all}"
    echo -e "\n${GREEN}═══ Validating $model_count models (filter: $filter_label) ═══${NC}"

    # Read registry and dispatch to run_add_model
    local _skip_until="$START_FROM"
    while IFS=$'\t' read -r MODEL_NAME ADD_TASK PP DXNN_FILE; do
        # --start-from: skip models until the specified model is reached
        if [ -n "$_skip_until" ]; then
            if [ "$MODEL_NAME" = "$_skip_until" ]; then
                _skip_until=""
            else
                TOTAL=$((TOTAL + 1))
                continue
            fi
        fi
        local model_path="$ASSET_DIR/$DXNN_FILE"
        run_add_model "$MODEL_NAME" "$ADD_TASK" "$PP" "$model_path"
    done < <(python3 -c "
import json
with open('$REGISTRY') as f:
    models = json.load(f)
ft = '$filter_task'
for m in models:
    if ft and m['add_model_task'] != ft:
        continue
    if not m.get('supported', True):
        continue
    print(f\"{m['model_name']}\t{m['add_model_task']}\t{m['postprocessor']}\t{m['dxnn_file']}\")
")
}

# ============================================================================
# Clean mode: remove generated packages (preserves reference models)
# ============================================================================

# Reference directories that must NOT be deleted during clean
REFERENCE_DIRS=(
    "object_detection/yolov5n"
    "object_detection/yolov7"
    "object_detection/yoloxs"
    "object_detection/yolov8n"
    "object_detection/yolov9t"
    "object_detection/yolov10n"
    "object_detection/yolov11n"
    "object_detection/yolov12n"
    "object_detection/yolo26n"
    "object_detection/ssdmv1"
    "object_detection/nanodet_repvgg"
    "object_detection/damoyolot"
    "object_detection/centernet_resnet18"
    "face_detection/scrfd500m"
    "face_detection/yolov5s_face"
    "face_detection/retinaface_mobilenet0_25_640"
    "pose_estimation/yolov5pose"
    "pose_estimation/yolov8s_pose"
    "obb_detection/yolo26n_obb"
    "classification/efficientnet_lite0"
    "semantic_segmentation/deeplabv3plusmobilenet"
    "semantic_segmentation/bisenetv1"
    "semantic_segmentation/segformer_b0_512x1024"
    "instance_segmentation/yolov8n_seg"
    "instance_segmentation/yolov5n_seg"
    "instance_segmentation/yolact_regnetx_800mf"
    "depth_estimation/fastdepth_1"
    "image_denoising/dncnn_15"
    "super_resolution/espcn_x4"
    "image_enhancement/zero_dce"
    "embedding/clip_resnet50_image_encoder_224x224"
    "embedding/clip_resnet50_text_encoder_77x512"
    "embedding/arcface_mobilefacenet"
    "ppu/yolov5s_ppu"
    "ppu/yolov7_ppu"
    "ppu/yolov5pose_ppu"
    "hand_landmark/handlandmarklite_1"
)

is_reference() {
    local cat_pkg="$1"
    for ref in "${REFERENCE_DIRS[@]}"; do
        if [ "$cat_pkg" = "$ref" ]; then
            return 0
        fi
    done
    return 1
}

do_clean() {
    TEST_PACKAGES=()
    local filter_task
    filter_task=$(filter_to_task "$FILTER")

    # Collect all model names from registry
    while IFS=$'\t' read -r MODEL_NAME ADD_TASK PP DXNN_FILE; do
        run_add_model "$MODEL_NAME" "$ADD_TASK" "$PP" "$ASSET_DIR/$DXNN_FILE"
    done < <(python3 -c "
import json
with open('$REGISTRY') as f:
    models = json.load(f)
ft = '$filter_task'
for m in models:
    if ft and m['add_model_task'] != ft:
        continue
    if not m.get('supported', True):
        continue
    print(f\"{m['model_name']}\t{m['add_model_task']}\t{m['postprocessor']}\t{m['dxnn_file']}\")
")

    echo ""
    echo "Cleaning ${#TEST_PACKAGES[@]} test packages ($CURRENT_LANG)..."
    echo "  (Reference models are preserved)"
    cd "$CURRENT_SRC_DIR"

    local removed=0 skipped=0
    if [ "$CURRENT_LANG" = "cpp" ]; then
        CMAKE="$CURRENT_SRC_DIR/CMakeLists.txt"
        REMOVER="$CURRENT_SRC_DIR/cmake_remove_model.py"
        for pkg in "${TEST_PACKAGES[@]}"; do
            dir=$(find . -maxdepth 2 -type d -name "$pkg" 2>/dev/null | head -1)
            if [ -n "$dir" ]; then
                local rel_path="${dir#./}"
                if is_reference "$rel_path"; then
                    skipped=$((skipped + 1))
                    continue
                fi
                python3 -c "import shutil; shutil.rmtree('$dir', ignore_errors=True)"
                echo "  Removed: $dir"
                removed=$((removed + 1))
            fi
            python3 "$REMOVER" "$CMAKE" "$pkg" 2>/dev/null || true
        done
        echo "Done. Removed: $removed, Preserved references: $skipped"
        echo "Run 'cd build && cmake ..' to reconfigure."
    else
        for pkg in "${TEST_PACKAGES[@]}"; do
            cat="${PKG_CATEGORIES[$pkg]}"
            dir="$CURRENT_SRC_DIR/$cat/$pkg"
            if is_reference "$cat/$pkg"; then
                skipped=$((skipped + 1))
                continue
            fi
            if [ -d "$dir" ]; then
                python3 -c "import shutil; shutil.rmtree('$dir', ignore_errors=True)"
                echo "  Removed: $cat/$pkg"
                removed=$((removed + 1))
            fi
        done
        echo "Done. Removed: $removed, Preserved references: $skipped"
    fi

    # Remove orphan directories (not in registry and not reference)
    local orphan_removed=0
    declare -A known_names
    for pkg in "${TEST_PACKAGES[@]}"; do
        known_names["$pkg"]=1
    done
    for ref in "${REFERENCE_DIRS[@]}"; do
        local ref_name="${ref##*/}"
        known_names["$ref_name"]=1
    done

    for cat_dir in "$CURRENT_SRC_DIR"/*/; do
        [ -d "$cat_dir" ] || continue
        local cat_name
        cat_name=$(basename "$cat_dir")
        [[ "$cat_name" == "common" ]] && continue
        for model_dir in "$cat_dir"*/; do
            [ -d "$model_dir" ] || continue
            local model_name
            model_name=$(basename "$model_dir")
            [[ "$model_name" == "__pycache__" ]] && { rm -rf "$model_dir"; orphan_removed=$((orphan_removed + 1)); continue; }
            if [ -z "${known_names[$model_name]+x}" ]; then
                rm -rf "$model_dir"
                echo "  Orphan removed: $cat_name/$model_name"
                orphan_removed=$((orphan_removed + 1))
            fi
        done
        # Remove empty category dirs
        if [ -d "$cat_dir" ] && [ -z "$(ls -A "$cat_dir")" ]; then
            rmdir "$cat_dir"
            echo "  Empty category removed: $cat_name"
        fi
    done
    if [ "$orphan_removed" -gt 0 ]; then
        echo "  Orphan cleanup: $orphan_removed removed"
    fi
}

# ============================================================================
# Main
# ============================================================================
# Validate filter before starting
filter_to_task "$FILTER" > /dev/null

REGISTRY_TOTAL=$(python3 -c "import json; print(len(json.load(open('$REGISTRY'))))")

for CURRENT_LANG in "${LANGS[@]}"; do
    case "$CURRENT_LANG" in
        cpp) CURRENT_SRC_DIR="$CPP_SRC_DIR" ;;
        py)  CURRENT_SRC_DIR="$PY_SRC_DIR" ;;
    esac

    LANG_LABEL=$(echo "$CURRENT_LANG" | tr '[:lower:]' '[:upper:]')

    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  DX App Model Validation ($LANG_LABEL)                          ║${NC}"
    echo -e "${CYAN}║  Registry: ${REGISTRY_TOTAL} models  |  Filter: ${FILTER}              ║${NC}"
    echo -e "${CYAN}║  Asset dir: $(basename $ASSET_DIR)                              ║${NC}"
    echo -e "${CYAN}║  Started: $(date '+%Y-%m-%d %H:%M:%S')                         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"

    if [ "$CLEAN_MODE" = true ]; then
        do_clean
    else
        FILTER_TASK=$(filter_to_task "$FILTER")
        run_from_registry "$FILTER_TASK"
    fi
done

if [ "$LIST_ONLY" = false ] && [ "$CLEAN_MODE" = false ]; then
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Validation Summary${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "  Registry: ${REGISTRY_TOTAL} models  |  Filter: ${FILTER}"
    echo -e "  Total: ${TOTAL}  ${GREEN}PASS: ${PASS}${NC}  ${RED}FAIL: ${FAIL}${NC}  ${YELLOW}SKIP: ${SKIP}${NC}  ${YELLOW}CONFIG_WARN: ${CONFIG_WARN}${NC}"
    if [ "$NUMERICAL" = true ]; then
        echo -e "  Numerical: ${GREEN}PASS: ${NUM_PASS}${NC}  ${RED}FAIL: ${NUM_FAIL}${NC}  ${YELLOW}WARN: ${NUM_WARN}${NC}  SKIP: ${NUM_SKIP}"
    fi
    echo -e "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "  Log dir: $LOG_DIR"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    if [ ${#RMAP_FAILED_MODELS[@]} -gt 0 ]; then
        rmap_log="$LOG_DIR/rmap_failed_models.txt"
        echo -e "  ${YELLOW}NPU rmap failed models (${#RMAP_FAILED_MODELS[@]}): $rmap_log${NC}"
        printf '%s\n' "${RMAP_FAILED_MODELS[@]}" > "$rmap_log"
    fi
    if [ ${#NUM_FAILED_MODELS[@]} -gt 0 ]; then
        echo -e "  ${RED}Numerical failed models:${NC}"
        for m in "${NUM_FAILED_MODELS[@]}"; do
            echo "    - $m"
        done
    fi
    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        echo -e "  ${RED}Failed models:${NC}"
        for m in "${FAILED_MODELS[@]}"; do
            echo "    - $m"
        done
    fi
    if [ ${#CONFIG_WARNINGS[@]} -gt 0 ]; then
        config_warn_log="$LOG_DIR/config_warnings.log"
        echo -e "  ${YELLOW}Config warnings:${NC}"
        for w in "${CONFIG_WARNINGS[@]}"; do
            echo "    ${SYM_WARN} $w"
        done
        printf '%s\n' "${CONFIG_WARNINGS[@]}" > "$config_warn_log"
        echo -e "  ${YELLOW}Config warnings saved: $config_warn_log${NC}"
    fi

    # Write summary log
    {
        echo "Validation Summary"
        echo "  Date:    $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Filter:  $FILTER"
        echo "  Lang:    $LANG_MODE"
        echo "  Total:   $TOTAL  PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP  CONFIG_WARN: $CONFIG_WARN"
        if [ "$NUMERICAL" = true ]; then
            echo "  Numerical: PASS: $NUM_PASS  FAIL: $NUM_FAIL  WARN: $NUM_WARN  SKIP: $NUM_SKIP"
            if [ ${#NUM_FAILED_MODELS[@]} -gt 0 ]; then
                echo "  Numerical failed models:"
                for m in "${NUM_FAILED_MODELS[@]}"; do
                    echo "    - $m"
                done
            fi
        fi
        if [ ${#RMAP_FAILED_MODELS[@]} -gt 0 ]; then
            echo "  NPU rmap failed models (${#RMAP_FAILED_MODELS[@]}):"
            for m in "${RMAP_FAILED_MODELS[@]}"; do
                echo "    - $m"
            done
        fi
        if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
            echo "  Failed models:"
            for m in "${FAILED_MODELS[@]}"; do
                echo "    - $m"
            done
        fi
        if [ ${#CONFIG_WARNINGS[@]} -gt 0 ]; then
            echo "  Config warnings (${#CONFIG_WARNINGS[@]}):"
            for w in "${CONFIG_WARNINGS[@]}"; do
                echo "    ! $w"
            done
        fi
    } | tee "$SUMMARY_LOG"
    echo "  Summary: $SUMMARY_LOG"
fi
