#!/bin/bash
# =============================================================================
# dx_tool.sh - DX-APP Unified Tool
# =============================================================================
# Provides a single interface for adding models, extracting standalone packages,
# listing/searching models, and more.
#
# Usage:
#   ./scripts/dx_tool.sh                  # Interactive menu
#   ./scripts/dx_tool.sh add              # Add new model
#   ./scripts/dx_tool.sh extract          # Extract standalone package
#   ./scripts/dx_tool.sh list [FILTER]    # List models (category filter)
#   ./scripts/dx_tool.sh search KEYWORD   # Search models
#   ./scripts/dx_tool.sh info MODEL       # Model detail info
#   ./scripts/dx_tool.sh delete MODEL     # Delete model
#   ./scripts/dx_tool.sh new-task         # Create new task skeleton
#   ./scripts/dx_tool.sh validate         # Validate all models
#   ./scripts/dx_tool.sh run [OPTS...]    # Run examples
#   ./scripts/dx_tool.sh bench [OPTS...]  # Benchmark models
#   ./scripts/dx_tool.sh help             # Help
#
# Enable tab completion:
#   source scripts/dx_tool_completion.bash
# =============================================================================

set -e

# Tab completion hint (shown once on interactive startup)
if [[ -t 1 && -z "${DX_TOOL_COMP_HINT_SHOWN:-}" ]]; then
    if ! complete -p dx_tool.sh &>/dev/null 2>&1 && \
       ! complete -p ./scripts/dx_tool.sh &>/dev/null 2>&1; then
        echo -e "\033[2m[Tip] Enable tab completion: source scripts/dx_tool_completion.bash\033[0m" >&2
    fi
    export DX_TOOL_COMP_HINT_SHOWN=1
fi

VERSION="1.1.0"

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_DIR="$PROJECT_ROOT/src/cpp_example"
PY_DIR="$PROJECT_ROOT/src/python_example"

# =============================================================================
# Colors & Symbols
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

SYM_OK="✓"
SYM_FAIL="✗"
SYM_WARN="⚠"
SYM_ARROW="→"
SYM_DOT="•"

# =============================================================================
# Utility Functions
# =============================================================================
# Print to stderr (safe for command substitution)
_msg()  { echo -e "$@" >&2; }
_info() { echo -e "${CYAN}${SYM_DOT}${NC} $*" >&2; }
_ok()   { echo -e "${GREEN}${SYM_OK}${NC} $*" >&2; }
_warn() { echo -e "${YELLOW}${SYM_WARN}${NC} $*" >&2; }
_err()  { echo -e "${RED}${SYM_FAIL}${NC} $*" >&2; }

# Developer-only password gate (options 1, 6, 7)
_DEV_PASSWORD="02230301"
_require_dev_password() {
    local action="${1:-this action}"
    local attempts=3
    while (( attempts-- > 0 )); do
        local input
        read -rsp "$(echo -e "${YELLOW}[DEV]${NC} Password required for ${action}: ")" input
        echo "" >&2
        if [[ "$input" == "$_DEV_PASSWORD" ]]; then
            return 0
        fi
        _err "Incorrect password. Attempts remaining: ${attempts}"
    done
    _err "Access denied."
    return 1
}

print_header() {
    local cpp_count py_count
    cpp_count=$(_count_models_fast cpp)
    py_count=$(_count_models_fast python)

    # Box inner width = 54 characters
    local W=54

    # Helper: print a box line with exact width padding
    # Usage: _bx "visible text" "ansi text"
    _bx() {
        local visible="$1" ansi="$2"
        local pad=$((W - ${#visible}))
        (( pad < 0 )) && pad=0
        printf "${BOLD}║${NC}%b%*s${BOLD}║${NC}\n" "$ansi" "$pad" ""
    }

    local border
    printf -v border '%*s' $W ''
    border=${border// /═}

    local info
    printf -v info 'C++ %s | Python %s' "$cpp_count" "$py_count"

    echo ""
    echo -e "${BOLD}╔${border}╗${NC}"
    _bx "  DX Model Tool  v${VERSION}" "  ${CYAN}DX Model Tool${NC}  ${DIM}v${VERSION}${NC}"
    _bx "  ${info}" "  ${DIM}${info}${NC}"
    echo -e "${BOLD}╠${border}╣${NC}"
    _bx "   1. Add Model              (add)" "  ${GREEN} 1${NC}. Add Model              ${CYAN}(add)${NC}"
    _bx "   2. Extract Package        (extract)" "  ${GREEN} 2${NC}. Extract Package        ${CYAN}(extract)${NC}"
    _bx "   3. List Models            (list)" "  ${GREEN} 3${NC}. List Models            ${CYAN}(list)${NC}"
    _bx "   4. Search                 (search)" "  ${GREEN} 4${NC}. Search                 ${CYAN}(search)${NC}"
    _bx "   5. Model Info             (info)" "  ${GREEN} 5${NC}. Model Info             ${CYAN}(info)${NC}"
    _bx "   6. Delete Model           (delete)" "  ${GREEN} 6${NC}. Delete Model           ${CYAN}(delete)${NC}"
    _bx "   7. New Task Skeleton      (new-task)" "  ${GREEN} 7${NC}. New Task Skeleton      ${CYAN}(new-task)${NC}"
    _bx "   8. Validate All           (validate)" "  ${GREEN} 8${NC}. Validate All           ${CYAN}(validate)${NC}"
    _bx "   9. Run Examples           (run)" "  ${GREEN} 9${NC}. Run Examples           ${CYAN}(run)${NC}"
    _bx "  10. Benchmark              (bench)" "  ${GREEN}10${NC}. Benchmark              ${CYAN}(bench)${NC}"
    _bx "   0. Exit" "  ${GREEN} 0${NC}. Exit"
    echo -e "${BOLD}╚${border}╝${NC}"
}

# Fast model count (for header) — counts models with actual sync files
_count_models_fast() {
    local lang="$1" dir count=0
    if [[ "$lang" == "cpp" ]]; then dir="$CPP_DIR"; else dir="$PY_DIR"; fi
    for cat_dir in "$dir"/*/; do
        [[ ! -d "$cat_dir" ]] && continue
        local cat_name=$(basename "$cat_dir")
        [[ "$cat_name" == "common" || "$cat_name" == "build" || "$cat_name" == "sample" || "$cat_name" == "__pycache__" || "$cat_name" == "utils" ]] && continue
        for model_dir in "$cat_dir"/*/; do
            [[ ! -d "$model_dir" ]] && continue
            local mname=$(basename "$model_dir")
            [[ "$mname" == "__pycache__" || "$mname" == "build" ]] && continue
            # Verify the model has an actual sync entry point
            if [[ "$lang" == "cpp" ]]; then
                ls "$model_dir"/*_sync.cpp &>/dev/null && count=$((count + 1))
            else
                ls "$model_dir"/*_sync.py &>/dev/null && count=$((count + 1))
            fi
        done
    done
    echo "$count"
}

# ─────────────────────────────────────────────────────────────────────────────
# Interactive readline prompt with tab completion support
# Usage:
#   _read_complete "prompt text" RESULT_VAR word1 word2 word3 ...
#   _read_complete "prompt text" RESULT_VAR       # no completions, just readline
#
# Uses read -e for readline + temporary COMP_WORDBREAKS binding.
# Pressing Tab cycles through matching completions.
# ─────────────────────────────────────────────────────────────────────────────
_read_complete() {
    local prompt="$1"
    local -n _rc_result="$2"
    shift 2
    local completions=("$@")

    if [[ ${#completions[@]} -gt 0 ]]; then
        # Install a temporary readline completion
        local _old_comp
        _old_comp=$(complete -p -E 2>/dev/null || true)

        _dx_complete_fn() {
            local cur="${COMP_WORDS[COMP_CWORD]}"
            COMPREPLY=( $(compgen -W "${_DX_COMP_WORDS}" -- "$cur") )
        }

        export _DX_COMP_WORDS="${completions[*]}"
        complete -F _dx_complete_fn -E
        complete -F _dx_complete_fn bash

        read -erp "$prompt" _rc_result

        # Restore original completion
        complete -r -E 2>/dev/null || true
        complete -r bash 2>/dev/null || true
        if [[ -n "$_old_comp" ]]; then
            eval "$_old_comp" 2>/dev/null || true
        fi
        unset _DX_COMP_WORDS
        unset -f _dx_complete_fn 2>/dev/null || true
    else
        read -erp "$prompt" _rc_result
    fi
}

# Build the list of all model names (for completion)
_get_all_model_names() {
    local names=()
    for base_dir in "$CPP_DIR" "$PY_DIR"; do
        for cat_dir in "$base_dir"/*/; do
            [[ ! -d "$cat_dir" ]] && continue
            local cat_name=$(basename "$cat_dir")
            [[ "$cat_name" == "common" || "$cat_name" == "build" || "$cat_name" == "sample" || "$cat_name" == "__pycache__" || "$cat_name" == "utils" ]] && continue
            for model_dir in "$cat_dir"/*/; do
                [[ ! -d "$model_dir" ]] && continue
                local mname=$(basename "$model_dir")
                [[ "$mname" == "__pycache__" || "$mname" == "build" ]] && continue
                names+=("$mname")
            done
        done
    done
    # Also add category/model format
    for base_dir in "$CPP_DIR" "$PY_DIR"; do
        for cat_dir in "$base_dir"/*/; do
            [[ ! -d "$cat_dir" ]] && continue
            local cat_name=$(basename "$cat_dir")
            [[ "$cat_name" == "common" || "$cat_name" == "build" || "$cat_name" == "sample" || "$cat_name" == "__pycache__" || "$cat_name" == "utils" ]] && continue
            for model_dir in "$cat_dir"/*/; do
                [[ ! -d "$model_dir" ]] && continue
                local mname=$(basename "$model_dir")
                [[ "$mname" == "__pycache__" || "$mname" == "build" ]] && continue
                names+=("$cat_name/$mname")
            done
        done
    done
    # Deduplicate
    printf '%s\n' "${names[@]}" | sort -u
}

# Build category→models map from test_models.conf
# Populates: _FILTER_CATS (ordered array), _FILTER_CAT_COUNTS (assoc), _FILTER_CAT_MODELS (assoc, space-separated)
_load_conf_categories() {
    local conf="${PROJECT_ROOT}/config/test_models.conf"
    _FILTER_CATS=()
    declare -gA _FILTER_CAT_COUNTS=()
    declare -gA _FILTER_CAT_MODELS=()
    [[ ! -f "$conf" ]] && return

    while IFS=$'\t' read -r name category _; do
        [[ -z "$name" || "$name" == \#* ]] && continue
        if [[ -z "${_FILTER_CAT_COUNTS[$category]+x}" ]]; then
            _FILTER_CATS+=("$category")
            _FILTER_CAT_COUNTS[$category]=0
            _FILTER_CAT_MODELS[$category]=""
        fi
        _FILTER_CAT_COUNTS[$category]=$(( ${_FILTER_CAT_COUNTS[$category]} + 1 ))
        _FILTER_CAT_MODELS[$category]+="$name "
    done < "$conf"
}

# Interactive filter prompt — shows categories, lets user pick scope
# Usage: _prompt_filter extra_args
_prompt_filter() {
    local -n _pf_args="$1"

    _load_conf_categories

    # Show category summary
    _msg ""
    _msg "${DIM}─── Available Categories ───${NC}"
    local total=0
    for cat in "${_FILTER_CATS[@]}"; do
        printf "  ${YELLOW}%-28s${NC} ${DIM}%3d models${NC}\n" "$cat" "${_FILTER_CAT_COUNTS[$cat]}" >&2
        total=$(( total + ${_FILTER_CAT_COUNTS[$cat]} ))
    done
    _msg "${DIM}  Total: ${total} models${NC}"

    local filter_choice
    filter_choice=$(prompt_select "Filter:" "All models (no filter)" "By category" "By keyword")

    case "$filter_choice" in
        1) ;; # no filter
        2)
            local -a cat_opts=()
            for cat in "${_FILTER_CATS[@]}"; do
                cat_opts+=("${cat}  (${_FILTER_CAT_COUNTS[$cat]})")
            done
            local cat_idx
            cat_idx=$(prompt_select "Select category:" "${cat_opts[@]}")
            local selected="${_FILTER_CATS[$((cat_idx-1))]}"
            _pf_args+=("--category" "$selected")

            # Show models in selected category
            _msg ""
            _msg "${DIM}Models in ${YELLOW}${selected}${NC}${DIM}:${NC}"
            local -a cat_model_arr=(${_FILTER_CAT_MODELS[$selected]})
            local col=0
            for m in "${cat_model_arr[@]}"; do
                printf "  %-25s" "$m" >&2
                col=$(( col + 1 ))
                (( col % 3 == 0 )) && echo "" >&2
            done
            (( col % 3 != 0 )) && echo "" >&2

            # Optional name filter within category
            _msg ""
            local name_filter
            _read_complete "$(echo -e "${CYAN}▸${NC} Name filter (Enter = all in category): ")" name_filter "${cat_model_arr[@]}"
            [[ -n "$name_filter" && "$name_filter" != "all" ]] && _pf_args+=("--filter" "$name_filter") || true
            ;;
        3)
            local filter
            local _model_names
            _model_names=($(_get_all_model_names))
            _read_complete "$(echo -e "${CYAN}▸${NC} Keyword (Tab = autocomplete): ")" filter "${_model_names[@]}"
            [[ -n "$filter" ]] && _pf_args+=("--filter" "$filter") || true
            ;;
    esac
}

# Selection prompt — menu on stderr. With retry.
prompt_select() {
    local prompt="$1"
    shift
    local options=("$@")
    local max=${#options[@]}
    while true; do
        _msg ""
        _msg "${BOLD}${prompt}${NC}"
        for i in "${!options[@]}"; do
            _msg "  ${GREEN}$((i+1))${NC}) ${options[$i]}"
        done
        _msg ""
        read -rp "Select [1-${max}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= max )); then
            echo "$choice"
            return 0
        fi
        _err "Please enter a number between 1 and ${max}"
    done
}

# Y/n confirmation prompt
confirm_prompt() {
    local msg="${1:-Continue?}"
    read -rp "$msg [Y/n]: " answer
    [[ ! "$answer" =~ ^[Nn] ]]
}

# Divider line
print_divider() {
    local title="$1"
    echo -e "\n${BOLD}━━━ ${title} ━━━${NC}"
}

# =============================================================================
# Task Types & Postprocessor Reference Data
# =============================================================================
TASK_TYPES=(
    "object_detection"
    "classification"
    "semantic_segmentation"
    "instance_segmentation"
    "pose_estimation"
    "face_detection"
    "obb_detection"
    "depth_estimation"
    "image_denoising"
    "super_resolution"
    "image_enhancement"
    "embedding"
    "ppu"
    "hand_landmark"
    "attribute_recognition"
    "reid"
)

TASK_DESCRIPTIONS=(
    "Object Detection (YOLOv5/v7/v8/v9/v10/v11/v12/v26, SSD, ...)"
    "Image Classification (EfficientNet, ResNet, ...)"
    "Semantic Segmentation (BiSeNet, DeepLabV3, SegFormer)"
    "Instance Segmentation (YOLOv5-Seg, YOLOv8-Seg, YOLACT)"
    "Pose Estimation (YOLOv5-Pose, YOLOv8-Pose)"
    "Face Detection (SCRFD, RetinaFace, YOLOv5-Face)"
    "Oriented Bounding Box (OBB)"
    "Depth Estimation (FastDepth, SCDepthV3)"
    "Image Denoising (DnCNN)"
    "Super Resolution (ESPCN)"
    "Image Enhancement (Zero-DCE)"
    "Embedding (ArcFace)"
    "Pre-Processing Unit (PPU)"
    "Hand Landmark (MediaPipe)"
    "Attribute Recognition (DeepMAR)"
    "Person Re-Identification (CasViT)"
)

# Category directory names
CATEGORIES=(
    classification object_detection semantic_segmentation
    instance_segmentation depth_estimation ppu
    embedding face_detection
    pose_estimation obb_detection super_resolution
    image_denoising image_enhancement
    hand_landmark attribute_recognition reid
)

# Task type -> default category mapping
# Now that task types match category names, this is mostly identity.
task_to_category() {
    case "$1" in
        object_detection)        echo "object_detection" ;;
        classification)          echo "classification" ;;
        semantic_segmentation)   echo "semantic_segmentation" ;;
        instance_segmentation)   echo "instance_segmentation" ;;
        pose_estimation)         echo "pose_estimation" ;;
        face_detection)          echo "face_detection" ;;
        obb_detection)           echo "obb_detection" ;;
        depth_estimation)        echo "depth_estimation" ;;
        image_denoising)         echo "image_denoising" ;;
        super_resolution)        echo "super_resolution" ;;
        image_enhancement)       echo "image_enhancement" ;;
        embedding)               echo "embedding" ;;
        ppu)                     echo "ppu" ;;
        hand_landmark)           echo "hand_landmark" ;;
        attribute_recognition)   echo "attribute_recognition" ;;
        reid)                    echo "reid" ;;
        *)                       echo "object_detection" ;;
    esac
}

# Task type -> available postprocessor references
task_to_postprocessors() {
    case "$1" in
        object_detection)        echo "yolov5 yolov7 yolov8 yolov9 yolov10 yolov11 yolov12 yolov26 yolox ssd nanodet damoyolo centernet" ;;
        classification)          echo "efficientnet" ;;
        semantic_segmentation)   echo "bisenetv1 bisenetv2 deeplabv3 segformer" ;;
        instance_segmentation)   echo "yolov5seg yolov8seg yolact" ;;
        pose_estimation)         echo "yolov5pose yolov8pose" ;;
        face_detection)          echo "scrfd retinaface yolov5face yolov7face" ;;
        obb_detection)           echo "obb" ;;
        depth_estimation)        echo "depth" ;;
        image_denoising)         echo "dncnn" ;;
        super_resolution)        echo "espcn" ;;
        image_enhancement)       echo "zero_dce" ;;
        embedding)               echo "arcface embedding" ;;
        ppu)                     echo "yolov5_ppu yolov7_ppu yolov8n_ppu yolov8s_ppu scrfd_ppu yolov5pose_ppu" ;;
        hand_landmark)           echo "hand_landmark" ;;
        attribute_recognition)   echo "classification" ;;
        reid)                    echo "embedding" ;;
    esac
}

# =============================================================================
# 1. Add Model (Interactive)
# =============================================================================
do_add_model() {
    _require_dev_password "Add Model" || return 1
    print_divider "Add New Model"

    # Language selection
    local lang_choice
    lang_choice=$(prompt_select "Select language:" "C++" "Python" "Both")
    case "$lang_choice" in
        1) LANG_MODE="cpp" ;;
        2) LANG_MODE="python" ;;
        3) LANG_MODE="both" ;;
    esac
    _info "Language: ${BOLD}${LANG_MODE}${NC}"

    # Task type selection (with descriptions)
    local task_opts=()
    for i in "${!TASK_TYPES[@]}"; do
        task_opts+=("${TASK_TYPES[$i]}  ${DIM}— ${TASK_DESCRIPTIONS[$i]}${NC}")
    done
    local task_choice
    task_choice=$(prompt_select "Select task type:" "${task_opts[@]}")
    local TASK_TYPE="${TASK_TYPES[$((task_choice-1))]}"
    local CATEGORY
    CATEGORY=$(task_to_category "$TASK_TYPE")
    _info "Task: ${BOLD}${TASK_TYPE}${NC}  Category: ${BOLD}${CATEGORY}${NC}"

    # Postprocessor (reference) selection
    local pp_list
    pp_list=$(task_to_postprocessors "$TASK_TYPE")
    local pp_array=($pp_list)

    local pp_choice
    pp_choice=$(prompt_select "Select postprocessor (reference):" "${pp_array[@]}")
    local POSTPROCESSOR="${pp_array[$((pp_choice-1))]}"
    _info "Postprocessor: ${BOLD}${POSTPROCESSOR}${NC}"

    # Model name input (validation + retry)
    local MODEL_NAME=""
    while true; do
        echo "" >&2
        read -rp "New model name (e.g. yolov30, my_detector): " MODEL_NAME
        MODEL_NAME=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
        if [[ -z "$MODEL_NAME" ]]; then
            _err "Model name is empty"; continue
        fi
        if [[ ! "$MODEL_NAME" =~ ^[a-z][a-z0-9_]*$ ]]; then
            _err "Model name must start with lowercase letter and only contain [a-z0-9_]"; continue
        fi
        # Duplicate check
        if [[ -d "$CPP_DIR/$CATEGORY/$MODEL_NAME" ]] || [[ -d "$PY_DIR/$CATEGORY/$MODEL_NAME" ]]; then
            _warn "Model already exists: ${CATEGORY}/${MODEL_NAME}"
            confirm_prompt "Overwrite?" || continue
        fi
        break
    done
    _info "Model name: ${BOLD}${MODEL_NAME}${NC}"

    # Sync/Async selection
    local mode_choice
    mode_choice=$(prompt_select "Generation mode:" "Sync + Async (default)" "Sync only")
    local SYNC_ONLY=""
    [[ "$mode_choice" == "2" ]] && SYNC_ONLY="--sync-only"

    # Final confirmation
    echo ""
    echo -e "${BOLD}┌─── Summary ───────────────────────────────┐${NC}"
    echo -e "${BOLD}│${NC}  Model:          ${CYAN}${MODEL_NAME}${NC}"
    echo -e "${BOLD}│${NC}  Task:           ${CYAN}${TASK_TYPE}${NC}"
    echo -e "${BOLD}│${NC}  Category:       ${CYAN}${CATEGORY}${NC}"
    echo -e "${BOLD}│${NC}  Postprocessor:  ${CYAN}${POSTPROCESSOR}${NC}"
    echo -e "${BOLD}│${NC}  Language:       ${CYAN}${LANG_MODE}${NC}"
    [[ -n "$SYNC_ONLY" ]] && echo -e "${BOLD}│${NC}  Mode:           ${YELLOW}Sync only${NC}"
    echo -e "${BOLD}└────────────────────────────────────────────┘${NC}"
    echo ""
    confirm_prompt "Proceed with generation?" || { echo "Cancelled"; return 0; }

    # Execute
    if [[ "$LANG_MODE" == "cpp" || "$LANG_MODE" == "both" ]]; then
        echo -e "\n${GREEN}[C++]${NC} Generating model..."
        bash "$SCRIPT_DIR/add_model.sh" "$MODEL_NAME" "$TASK_TYPE" --lang cpp \
            --category "$CATEGORY" --postprocessor "$POSTPROCESSOR" $SYNC_ONLY
        echo -e "${GREEN}[C++]${NC} Done: ${CYAN}src/cpp_example/${CATEGORY}/${MODEL_NAME}/${NC}"
    fi

    if [[ "$LANG_MODE" == "python" || "$LANG_MODE" == "both" ]]; then
        echo -e "\n${GREEN}[Python]${NC} Generating model..."
        bash "$SCRIPT_DIR/add_model.sh" "$MODEL_NAME" "$TASK_TYPE" --lang py \
            --category "$CATEGORY" --postprocessor "$POSTPROCESSOR" $SYNC_ONLY
        echo -e "${GREEN}[Python]${NC} Done: ${CYAN}src/python_example/${CATEGORY}/${MODEL_NAME}/${NC}"
    fi

    echo -e "\n${GREEN}✓ Model added successfully!${NC}"
    echo -e "  Build: ${YELLOW}cd src/cpp_example/build && cmake .. && make ${MODEL_NAME}_sync${NC}"
}

# =============================================================================
# 2. Extract Model Package
# =============================================================================
do_extract_model_package() {
    print_divider "Extract Standalone Package"

    local lang_choice
    lang_choice=$(prompt_select "Select language:" "C++" "Python" "Both")

    # Model path: interactive or direct input
    echo "" >&2
    echo -e "${DIM}Available categories: ${CATEGORIES[*]}${NC}" >&2
    echo -e "${DIM}Use Tab for completion. Enter 'all' for all models.${NC}" >&2
    local MODEL_PATH=""
    local _model_names
    _model_names=($(_get_all_model_names) "all")
    while true; do
        _read_complete "Model path (e.g. object_detection/yolov8): " MODEL_PATH "${_model_names[@]}"
        if [[ -z "$MODEL_PATH" ]]; then
            _err "Model path is empty"; continue
        fi
        if [[ "$MODEL_PATH" != "all" ]]; then
            # Validate existence
            if [[ ! -d "$CPP_DIR/$MODEL_PATH" ]] && [[ ! -d "$PY_DIR/$MODEL_PATH" ]]; then
                _err "Model not found: $MODEL_PATH"
                _info "To see available models: ${CYAN}dx_tool.sh list${NC}"
                continue
            fi
        fi
        break
    done

    # Output directory (optional)
    local OUTPUT_OPT=""
    echo "" >&2
    echo -e "${DIM}Examples: output, /tmp/standalone, my_package${NC}" >&2
    read -rp "Output directory (Enter = output): " OUTPUT_OPT
    OUTPUT_OPT="${OUTPUT_OPT## }"  # trim leading spaces
    OUTPUT_OPT="${OUTPUT_OPT%% }"  # trim trailing spaces
    # Default to 'output' if empty
    if [[ -z "$OUTPUT_OPT" ]]; then
        OUTPUT_OPT="output"
    fi

    case "$lang_choice" in
        1|3)
            echo -e "\n${GREEN}[C++]${NC} Extracting standalone package..."
            bash "$SCRIPT_DIR/extract_model_package.sh" "$MODEL_PATH" --lang cpp ${OUTPUT_OPT:+--output-dir "$OUTPUT_OPT"}
            echo -e "${GREEN}[C++]${NC} Done"
            ;;&
        2|3)
            echo -e "\n${GREEN}[Python]${NC} Extracting standalone package..."
            bash "$SCRIPT_DIR/extract_model_package.sh" "$MODEL_PATH" --lang py ${OUTPUT_OPT:+--output-dir "$OUTPUT_OPT"}
            echo -e "${GREEN}[Python]${NC} Done"
            ;;
    esac

    echo -e "\n${GREEN}✓ Standalone package extraction complete!${NC}"
}

# =============================================================================
# 3. List Models (Filter / Table View)
# =============================================================================
do_list_models() {
    local filter="${1:-}"
    print_divider "Model List${filter:+ (filter: $filter)}"

    local cpp_total=0
    local py_total=0

    # Table header
    printf "  ${BOLD}%-30s %6s %8s %s${NC}\n" "Category/Model" "C++" "Python" "Postprocessor"
    echo -e "  ${DIM}$(printf '%.0s─' {1..70})${NC}"

    for cat in "${CATEGORIES[@]}"; do
        # Filtering
        [[ -n "$filter" && "$cat" != *"$filter"* ]] && continue

        local cpp_models=()
        local py_models=()

        # C++ models
        if [[ -d "$CPP_DIR/$cat" ]]; then
            while IFS= read -r d; do
                local name=$(basename "$d")
                if [[ -f "$d/${name}_sync.cpp" ]]; then
                    cpp_models+=("$name")
                fi
            done < <(find "$CPP_DIR/$cat" -mindepth 1 -maxdepth 1 -type d | sort)
        fi

        # Python models
        if [[ -d "$PY_DIR/$cat" ]]; then
            while IFS= read -r d; do
                local name=$(basename "$d")
                if [[ -f "$d/${name}_sync.py" ]]; then
                    py_models+=("$name")
                fi
            done < <(find "$PY_DIR/$cat" -mindepth 1 -maxdepth 1 -type d | sort)
        fi

        local count=${#cpp_models[@]}
        [[ ${#py_models[@]} -gt $count ]] && count=${#py_models[@]}

        if [[ $count -gt 0 ]]; then
            echo -e "\n  ${BOLD}${CYAN}${cat}${NC} ${DIM}(C++: ${#cpp_models[@]}, Python: ${#py_models[@]})${NC}"

            # All models (union) sorted
            local all_models=($(echo "${cpp_models[*]} ${py_models[*]}" | tr ' ' '\n' | sort -u))
            for m in "${all_models[@]}"; do
                local cpp_mark="${DIM}-${NC}"
                local py_mark="${DIM}-${NC}"
                [[ " ${cpp_models[*]} " =~ " $m " ]] && cpp_mark="${GREEN}${SYM_OK}${NC}"
                [[ " ${py_models[*]} " =~ " $m " ]] && py_mark="${GREEN}${SYM_OK}${NC}"

                # Extract postprocessor type from C++ factory
                local pp_type="${DIM}—${NC}"
                local factory_dir="$CPP_DIR/$cat/$m/factory"
                if [[ -d "$factory_dir" ]]; then
                    local pp_header
                    pp_header=$(grep -rh '#include.*postprocessor' "$factory_dir"/ 2>/dev/null | head -1 | sed 's/.*\///' | sed 's/\.hpp.*//' || true)
                    [[ -n "$pp_header" ]] && pp_type="${MAGENTA}${pp_header}${NC}"
                fi

                printf "    %-28s  %b    %b    %b\n" "$m" "$cpp_mark" "$py_mark" "$pp_type"
            done
            cpp_total=$((cpp_total + ${#cpp_models[@]}))
            py_total=$((py_total + ${#py_models[@]}))
        fi
    done

    # Multi-model
    if [[ -d "$CPP_DIR" ]]; then
        local multi_cats
        multi_cats=$(find "$CPP_DIR" -maxdepth 1 -type d -name "*_x_*" 2>/dev/null || true)
        for mc in $multi_cats; do
            local cat_name=$(basename "$mc")
            [[ -n "$filter" && "$cat_name" != *"$filter"* ]] && continue
            local mc_count=0
            echo -e "\n  ${BOLD}${MAGENTA}${cat_name}${NC} ${DIM}(multi-model)${NC}"
            for d in "$mc"/*/; do
                [[ -d "$d" ]] && echo "    $(basename "$d")" && mc_count=$((mc_count + 1))
            done
            cpp_total=$((cpp_total + mc_count))
        done
    fi

    echo -e "\n  ${BOLD}Total: C++ ${GREEN}${cpp_total}${NC}, Python ${GREEN}${py_total}${NC}${NC}"

    # Postprocessor reference
    echo ""
    print_divider "Available Postprocessor References"
    printf "  ${BOLD}%-25s %s${NC}\n" "Task" "Postprocessors"
    echo -e "  ${DIM}$(printf '%.0s─' {1..60})${NC}"
    for i in "${!TASK_TYPES[@]}"; do
        local task="${TASK_TYPES[$i]}"
        local pps
        pps=$(task_to_postprocessors "$task")
        printf "  %-25s ${CYAN}%s${NC}\n" "$task" "$pps"
    done
}

# =============================================================================
# 4. Search Models
# =============================================================================
do_search() {
    local keyword="${1:-}"
    if [[ -z "$keyword" ]]; then
        local _model_names
        _model_names=($(_get_all_model_names))
        _read_complete "Search keyword: " keyword "${_model_names[@]}"
    fi
    if [[ -z "$keyword" ]]; then
        _err "Keyword is empty"; return 1
    fi

    print_divider "Search: '$keyword'"

    local found=0

    for lang_label in "C++" "Python"; do
        local base_dir="$CPP_DIR"
        [[ "$lang_label" == "Python" ]] && base_dir="$PY_DIR"

        for cat_dir in "$base_dir"/*/; do
            [[ ! -d "$cat_dir" ]] && continue
            local cat_name=$(basename "$cat_dir")
            [[ "$cat_name" == "common" || "$cat_name" == "build" || "$cat_name" == "sample" ]] && continue

            for model_dir in "$cat_dir"/*/; do
                [[ ! -d "$model_dir" ]] && continue
                local model_name=$(basename "$model_dir")
                if [[ "$model_name" == *"$keyword"* ]] || [[ "$cat_name" == *"$keyword"* ]]; then
                    local color="$GREEN"
                    [[ "$lang_label" == "Python" ]] && color="$BLUE"
                    printf "  ${color}[%-6s]${NC}  %-25s  %s\n" "$lang_label" "$cat_name" "$model_name"
                    found=$((found + 1))
                fi
            done
        done
    done

    if [[ $found -eq 0 ]]; then
        _warn "No models matching '$keyword'"
    else
        echo -e "\n  ${BOLD}${found} result(s)${NC}"
    fi
}

# =============================================================================
# 5. Model Info
# =============================================================================
do_info() {
    local model_query="${1:-}"
    if [[ -z "$model_query" ]]; then
        local _model_names
        _model_names=($(_get_all_model_names))
        _read_complete "Model name (e.g. yolov8) or path (e.g. object_detection/yolov8): " model_query "${_model_names[@]}"
    fi
    if [[ -z "$model_query" ]]; then
        _err "Model name is empty"; return 1
    fi

    print_divider "Model Info: $model_query"

    # Model search (name-only searches all categories)
    local found_dirs=()
    for base_dir in "$CPP_DIR" "$PY_DIR"; do
        local lang="C++"
        [[ "$base_dir" == "$PY_DIR" ]] && lang="Python"

        if [[ "$model_query" == */* ]]; then
            # Path format
            if [[ -d "$base_dir/$model_query" ]]; then
                found_dirs+=("$lang:$base_dir/$model_query")
            fi
        else
            # Name-only: search all categories
            for cat_dir in "$base_dir"/*/; do
                [[ ! -d "$cat_dir" ]] && continue
                local cat_name=$(basename "$cat_dir")
                [[ "$cat_name" == "common" || "$cat_name" == "build" || "$cat_name" == "sample" ]] && continue
                if [[ -d "$cat_dir/$model_query" ]]; then
                    found_dirs+=("$lang:$cat_dir$model_query")
                fi
            done
        fi
    done

    if [[ ${#found_dirs[@]} -eq 0 ]]; then
        _err "Model not found: $model_query"
        _info "Try searching: ${CYAN}dx_tool.sh search ${model_query}${NC}"
        return 1
    fi

    for entry in "${found_dirs[@]}"; do
        local lang="${entry%%:*}"
        local dir="${entry#*:}"
        local model_name=$(basename "$dir")
        local cat_name=$(basename "$(dirname "$dir")")

        echo -e "\n  ${BOLD}[${lang}]${NC} ${CYAN}${cat_name}/${model_name}${NC}"
        echo -e "  ${DIM}$(printf '%.0s─' {1..50})${NC}"

        # File list
        echo -e "  ${BOLD}Files:${NC}"
        while IFS= read -r f; do
            local fname=$(basename "$f")
            local fsize=$(du -h "$f" 2>/dev/null | cut -f1)
            local lines=$(wc -l < "$f" 2>/dev/null || echo "?")
            printf "    %-35s %6s  %5s lines\n" "$fname" "$fsize" "$lines"
        done < <(find "$dir" -type f | sort)

        # Total directory size
        local total_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo -e "  ${BOLD}Total size:${NC} ${total_size}"

        # C++ factory info
        if [[ "$lang" == "C++" && -d "$dir/factory" ]]; then
            echo -e "  ${BOLD}Factory:${NC}"
            local factory_hpp=$(find "$dir/factory" -name "*.hpp" -type f | head -1)
            if [[ -n "$factory_hpp" ]]; then
                local class_name
                class_name=$(grep -oP 'class \K\w+(?=\s*:)' "$factory_hpp" 2>/dev/null | head -1 || echo "?")
                echo -e "    Class: ${MAGENTA}${class_name}${NC}"

                # Postprocessor in use
                local pp_include
                pp_include=$(grep '#include.*postprocessor' "$factory_hpp" 2>/dev/null | sed 's/.*\///' | sed 's/"//' || true)
                [[ -n "$pp_include" ]] && echo -e "    Postprocessor: ${CYAN}${pp_include}${NC}"

                # Visualizer in use
                local vis_include
                vis_include=$(grep '#include.*visualizer' "$factory_hpp" 2>/dev/null | sed 's/.*\///' | sed 's/"//' || true)
                [[ -n "$vis_include" ]] && echo -e "    Visualizer: ${CYAN}${vis_include}${NC}"
            fi
        fi

        # Python factory info
        if [[ "$lang" == "Python" && -d "$dir/factory" ]]; then
            echo -e "  ${BOLD}Factory:${NC}"
            local factory_py=$(find "$dir/factory" -name "*.py" ! -name "__init__*" -type f | head -1)
            if [[ -n "$factory_py" ]]; then
                local class_name
                class_name=$(grep -oP 'class \K\w+' "$factory_py" 2>/dev/null | head -1 || echo "?")
                echo -e "    Class: ${MAGENTA}${class_name}${NC}"

                local imports
                imports=$(grep -P 'from.*import|import.*postprocessor|import.*visualizer' "$factory_py" 2>/dev/null | head -5)
                if [[ -n "$imports" ]]; then
                    echo -e "    Imports:"
                    echo "$imports" | while read -r line; do
                        echo -e "      ${DIM}${line}${NC}"
                    done
                fi
            fi
        fi

        # config.json content display
        local config_file="$dir/config.json"
        if [[ -f "$config_file" ]]; then
            echo -e "  ${BOLD}config.json:${NC}"
            local config_content
            config_content=$(cat "$config_file" 2>/dev/null)
            if [[ -z "$config_content" || "$config_content" == "{}" ]]; then
                echo -e "    ${DIM}(empty)${NC}"
            else
                # Parse JSON key-value into sorted table
                if command -v python3 &>/dev/null; then
                    python3 -c "
import json, sys
try:
    cfg = json.loads('''$config_content''')
    if not cfg:
        print('    \033[2m(empty)\033[0m')
    else:
        max_key = max(len(k) for k in cfg)
        for k, v in cfg.items():
            vtype = type(v).__name__
            if isinstance(v, bool):
                vstr = 'true' if v else 'false'
            elif isinstance(v, float):
                vstr = f'{v:g}'
            else:
                vstr = str(v)
            print(f'    \033[0;36m{k:<{max_key}}\033[0m : {vstr}  \033[2m({vtype})\033[0m')
except Exception as e:
    print(f'    \033[0;31mParse error: {e}\033[0m')
" 2>/dev/null
                else
                    # raw output if python3 unavailable
                    echo -e "    ${DIM}${config_content}${NC}"
                fi
            fi
        else
            echo -e "  ${BOLD}config.json:${NC} ${DIM}N/A${NC}"
        fi
    done
}

# =============================================================================
# 6. Delete Model
# =============================================================================
do_delete_model() {
    _require_dev_password "Delete Model" || return 1
    local model_query="${1:-}"
    if [[ -z "$model_query" ]]; then
        echo -e "${DIM}Examples: yolov8, object_detection/yolov8, yolov5_custom${NC}" >&2
        echo -e "${DIM}Use Tab for completion.${NC}" >&2
        local _model_names
        _model_names=($(_get_all_model_names))
        _read_complete "Model name or path to delete: " model_query "${_model_names[@]}"
    fi
    if [[ -z "$model_query" ]]; then
        _err "Model name is empty"; return 1
    fi

    print_divider "Delete Model: $model_query"

    # Search for deletion targets
    local targets=()
    for base_dir in "$CPP_DIR" "$PY_DIR"; do
        local lang="C++"
        [[ "$base_dir" == "$PY_DIR" ]] && lang="Python"

        if [[ "$model_query" == */* ]]; then
            if [[ -d "$base_dir/$model_query" ]]; then
                targets+=("$lang:$base_dir/$model_query")
            fi
        else
            for cat_dir in "$base_dir"/*/; do
                [[ ! -d "$cat_dir" ]] && continue
                local cat_name=$(basename "$cat_dir")
                [[ "$cat_name" == "common" || "$cat_name" == "build" || "$cat_name" == "sample" ]] && continue
                if [[ -d "$cat_dir/$model_query" ]]; then
                    targets+=("$lang:$cat_dir$model_query")
                fi
            done
        fi
    done

    if [[ ${#targets[@]} -eq 0 ]]; then
        _err "Model not found: $model_query"
        return 1
    fi

    # Show deletion targets
    echo -e "\n  ${BOLD}Targets:${NC}"
    for entry in "${targets[@]}"; do
        local lang="${entry%%:*}"
        local dir="${entry#*:}"
        local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        local rel_path="${dir#$PROJECT_ROOT/}"
        echo -e "    ${RED}${SYM_FAIL}${NC} [${lang}] ${rel_path} ${DIM}(${size})${NC}"
    done

    # Select scope if multiple targets
    if [[ ${#targets[@]} -gt 1 ]]; then
        local del_choice
        del_choice=$(prompt_select "Delete scope:" "Delete all" "C++ only" "Python only" "Cancel")
        case "$del_choice" in
            4) echo "Cancelled"; return 0 ;;
        esac
    fi

    echo ""
    _warn "This action cannot be undone!"
    confirm_prompt "Are you sure you want to delete?" || { echo "Cancelled"; return 0; }

    for entry in "${targets[@]}"; do
        local lang="${entry%%:*}"
        local dir="${entry#*:}"
        local rel_path="${dir#$PROJECT_ROOT/}"

        # Language filter
        if [[ ${#targets[@]} -gt 1 ]]; then
            case "${del_choice:-1}" in
                2) [[ "$lang" == "Python" ]] && continue ;;
                3) [[ "$lang" == "C++" ]] && continue ;;
            esac
        fi

        rm -rf "$dir"
        _ok "Deleted: ${rel_path}"
    done

    echo ""
    _ok "Model deletion complete"
    _info "CMakeLists.txt uses auto-discovery, so no manual update is needed"
}

# =============================================================================
# 7. New Task Skeleton
# =============================================================================
do_new_task() {
    _require_dev_password "New Task Skeleton" || return 1
    print_divider "Create New Task Skeleton"
    echo -e "${YELLOW}Note: Use this to add an entirely new task type that does not exist yet.${NC}"
    echo -e "      To add a new model to an existing task, use '1. Add Model'.\n"

    read -rp "New task name (e.g. transformer, 3d_reconstruction): " TASK_NAME
    if [[ -z "$TASK_NAME" ]]; then
        echo -e "${RED}Task name is empty${NC}"; return 1
    fi
    TASK_NAME=$(echo "$TASK_NAME" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

    local lang_choice
    lang_choice=$(prompt_select "Select language:" "C++" "Python" "Both")

    echo -e "\n${BOLD}Files to generate:${NC}"

    # C++ skeleton
    if [[ "$lang_choice" == "1" || "$lang_choice" == "3" ]]; then
        local TASK_UPPER=$(echo "$TASK_NAME" | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)}1' | tr -d ' ')
        local CPP_PROC_DIR="$CPP_DIR/common/processors"
        local CPP_RUNNER_DIR="$CPP_DIR/common/runner"
        local CPP_VIS_DIR="$CPP_DIR/common/visualizers"
        local CPP_BASE_DIR="$CPP_DIR/common/base"

        echo -e "  ${CYAN}[C++]${NC}"
        echo -e "    common/processors/${TASK_NAME}_postprocessor.hpp"
        echo -e "    common/runner/sync_${TASK_NAME}_runner.hpp"
        echo -e "    common/runner/async_${TASK_NAME}_runner.hpp"
        echo -e "    common/visualizers/${TASK_NAME}_visualizer.hpp"
        echo -e "    common/base/i_${TASK_NAME}_factory.hpp"

        echo ""
        confirm_prompt "Generate files?" || { echo "Cancelled"; return 0; }

        # --- Factory Interface ---
        cat > "$CPP_BASE_DIR/i_${TASK_NAME}_factory.hpp" << FACTORY_EOF
/**
 * @file i_${TASK_NAME}_factory.hpp
 * @brief ${TASK_UPPER} Abstract Factory interface
 *
 * AUTO-GENERATED by dx_tool.sh new-task
 * Search "TODO" for implementation points.
 */
#pragma once
#include <memory>
#include "common/processors/${TASK_NAME}_postprocessor.hpp"
#include "common/visualizers/${TASK_NAME}_visualizer.hpp"
#include "common/inputs/preprocessor.hpp"

class I${TASK_UPPER}Factory {
public:
    virtual ~I${TASK_UPPER}Factory() = default;

    virtual std::unique_ptr<IPreprocessor> createPreprocessor() = 0;

    virtual std::unique_ptr<IPostprocessor> createPostprocessor() = 0;

    // TODO: Define result visualization interface
    virtual std::unique_ptr<${TASK_UPPER}Visualizer> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
};
FACTORY_EOF
        echo -e "    ${GREEN}✓${NC} i_${TASK_NAME}_factory.hpp"

        # --- Postprocessor ---
        cat > "$CPP_PROC_DIR/${TASK_NAME}_postprocessor.hpp" << POST_EOF
/**
 * @file ${TASK_NAME}_postprocessor.hpp
 * @brief ${TASK_UPPER} Postprocessor
 *
 * AUTO-GENERATED by dx_tool.sh new-task
 * Search "TODO" for implementation points.
 */
#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "dx/dxrt.h"

// TODO: Define result structure for ${TASK_NAME}
struct ${TASK_UPPER}Result {
    // TODO: Add fields specific to ${TASK_NAME} outputs
    // Example:
    //   float confidence;
    //   cv::Mat output_map;
};

class ${TASK_UPPER}Postprocessor {
public:
    ${TASK_UPPER}Postprocessor(int input_w = 640, int input_h = 640)
        : input_w_(input_w), input_h_(input_h) {}

    virtual ~${TASK_UPPER}Postprocessor() = default;

    // TODO: Implement postprocess() — parse model output tensors into ${TASK_UPPER}Result
    virtual std::vector<${TASK_UPPER}Result> postprocess(
        const std::vector<dx_tensor_t>& output_tensors,
        int orig_w, int orig_h) {
        (void)output_tensors;
        (void)orig_w;
        (void)orig_h;
        // TODO: Parse output tensors here
        return {};
    }

    // TODO: Return NPU output tensor names for this model
    virtual std::vector<std::string> get_npu_output_names() const {
        // TODO: Return actual tensor names
        return {"output"};
    }

    // TODO: Return CPU (ORT) output tensor names for this model
    virtual std::vector<std::string> get_cpu_output_names() const {
        // TODO: Return actual tensor names
        return {"output"};
    }

protected:
    int input_w_;
    int input_h_;
};
POST_EOF
        echo -e "    ${GREEN}✓${NC} ${TASK_NAME}_postprocessor.hpp"

        # --- Visualizer ---
        cat > "$CPP_VIS_DIR/${TASK_NAME}_visualizer.hpp" << VIS_EOF
/**
 * @file ${TASK_NAME}_visualizer.hpp
 * @brief ${TASK_UPPER} result visualizer
 *
 * AUTO-GENERATED by dx_tool.sh new-task
 * Search "TODO" for implementation points.
 */
#pragma once
#include <opencv2/opencv.hpp>
#include "common/processors/${TASK_NAME}_postprocessor.hpp"

class ${TASK_UPPER}Visualizer {
public:
    virtual ~${TASK_UPPER}Visualizer() = default;

    // TODO: Implement draw() — visualize ${TASK_UPPER}Result on the frame
    virtual cv::Mat draw(const cv::Mat& frame,
                         const std::vector<${TASK_UPPER}Result>& results) {
        cv::Mat output = frame.clone();
        // TODO: Draw results on output image
        // Example:
        //   for (const auto& r : results) {
        //       cv::putText(output, "result", cv::Point(10, 30), ...);
        //   }
        return output;
    }
};
VIS_EOF
        echo -e "    ${GREEN}✓${NC} ${TASK_NAME}_visualizer.hpp"

        # --- Sync Runner ---
        cat > "$CPP_RUNNER_DIR/sync_${TASK_NAME}_runner.hpp" << SYNC_EOF
/**
 * @file sync_${TASK_NAME}_runner.hpp
 * @brief Synchronous ${TASK_UPPER} inference runner
 *
 * AUTO-GENERATED by dx_tool.sh new-task
 * Search "TODO" for implementation points.
 */
#pragma once
#include <string>
#include <memory>
#include "common/base/i_${TASK_NAME}_factory.hpp"

// TODO: Implement the sync runner following the pattern in sync_detection_runner.hpp
// Key steps:
//   1. factory->createPreprocessor()
//   2. factory->createPostprocessor()
//   3. factory->createVisualizer()
//   4. Loop: preprocess → inference → postprocess → visualize

class Sync${TASK_UPPER}Runner {
public:
    static int run(I${TASK_UPPER}Factory& factory, int argc, char* argv[]) {
        (void)factory; (void)argc; (void)argv;
        // TODO: Implement sync inference loop
        // See common/runner/sync_detection_runner.hpp for reference
        return 0;
    }
};
SYNC_EOF
        echo -e "    ${GREEN}✓${NC} sync_${TASK_NAME}_runner.hpp"

        # --- Async Runner ---
        cat > "$CPP_RUNNER_DIR/async_${TASK_NAME}_runner.hpp" << ASYNC_EOF
/**
 * @file async_${TASK_NAME}_runner.hpp
 * @brief Asynchronous ${TASK_UPPER} inference runner
 *
 * AUTO-GENERATED by dx_tool.sh new-task
 * Search "TODO" for implementation points.
 */
#pragma once
#include <string>
#include <memory>
#include <thread>
#include "common/base/i_${TASK_NAME}_factory.hpp"

// TODO: Implement the async runner following the pattern in async_detection_runner.hpp
// Key differences from sync:
//   - Producer thread: preprocess + enqueue
//   - Consumer thread: dequeue + postprocess + visualize

class Async${TASK_UPPER}Runner {
public:
    static int run(I${TASK_UPPER}Factory& factory, int argc, char* argv[]) {
        (void)factory; (void)argc; (void)argv;
        // TODO: Implement async inference loop
        // See common/runner/async_detection_runner.hpp for reference
        return 0;
    }
};
ASYNC_EOF
        echo -e "    ${GREEN}✓${NC} async_${TASK_NAME}_runner.hpp"
    fi

    # Python skeleton
    if [[ "$lang_choice" == "2" || "$lang_choice" == "3" ]]; then
        local PY_PROC_DIR="$PY_DIR/common/processors"
        local PY_VIS_DIR="$PY_DIR/common/visualizers"
        local PY_RUNNER_DIR="$PY_DIR/common/runner"
        local TASK_CAMEL=$(echo "$TASK_NAME" | sed 's/_\(.\)/\U\1/g; s/^\(.\)/\U\1/')

        echo -e "  ${CYAN}[Python]${NC}"
        echo -e "    common/processors/${TASK_NAME}_postprocessor.py"
        echo -e "    common/visualizers/${TASK_NAME}_visualizer.py"

        [[ "$lang_choice" == "2" ]] && {
            echo ""
            confirm_prompt "Generate files?" || { echo "Cancelled"; return 0; }
        }

        # --- Python Postprocessor ---
        cat > "$PY_PROC_DIR/${TASK_NAME}_postprocessor.py" << PYPOST_EOF
"""
${TASK_NAME}_postprocessor.py - ${TASK_CAMEL} Postprocessor

AUTO-GENERATED by dx_tool.sh new-task
Search "TODO" for implementation points.
"""
import numpy as np
from typing import List, Dict, Any


class ${TASK_CAMEL}Postprocessor:
    """${TASK_CAMEL} postprocessor.

    TODO: Implement the postprocess() method to parse model outputs.
    """

    def __init__(self, input_width: int = 640, input_height: int = 640):
        self.input_width = input_width
        self.input_height = input_height

    def postprocess(self, outputs: List[np.ndarray],
                    orig_width: int, orig_height: int) -> List[Dict[str, Any]]:
        """Parse model output tensors into structured results.

        Args:
            outputs: List of numpy arrays from model inference
            orig_width: Original image width
            orig_height: Original image height

        Returns:
            List of result dicts

        TODO: Implement output tensor parsing logic
        """
        # TODO: Parse output tensors here
        # Example:
        #   raw = outputs[0]  # shape: [1, N, ...]
        #   results = []
        #   for i in range(raw.shape[1]):
        #       results.append({"score": float(raw[0, i, 4])})
        #   return results
        return []

    def get_output_names(self) -> List[str]:
        """Return model output tensor names.

        TODO: Return actual tensor names for this model.
        """
        return ["output"]
PYPOST_EOF
        echo -e "    ${GREEN}✓${NC} ${TASK_NAME}_postprocessor.py"

        # --- Python Visualizer ---
        cat > "$PY_VIS_DIR/${TASK_NAME}_visualizer.py" << PYVIS_EOF
"""
${TASK_NAME}_visualizer.py - ${TASK_CAMEL} Visualizer

AUTO-GENERATED by dx_tool.sh new-task
Search "TODO" for implementation points.
"""
import cv2
import numpy as np
from typing import List, Dict, Any


class ${TASK_CAMEL}Visualizer:
    """Visualize ${TASK_CAMEL} results on images.

    TODO: Implement the draw() method.
    """

    def draw(self, frame: np.ndarray,
             results: List[Dict[str, Any]]) -> np.ndarray:
        """Draw results on the frame.

        Args:
            frame: Input image (BGR)
            results: List of result dicts from postprocessor

        Returns:
            Annotated image

        TODO: Implement visualization logic
        """
        output = frame.copy()
        # TODO: Draw results on output image
        # Example:
        #   for r in results:
        #       cv2.putText(output, f"score: {r['score']:.2f}",
        #                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return output
PYVIS_EOF
        echo -e "    ${GREEN}✓${NC} ${TASK_NAME}_visualizer.py"
    fi

    echo -e "\n${GREEN}✓ New task '${TASK_NAME}' skeleton created!${NC}"
    echo -e "\n${BOLD}Next steps:${NC}"
    echo -e "  1. Search for ${YELLOW}TODO${NC} in generated files and implement"
    echo -e "  2. After implementing Runner and Factory, add models: ${CYAN}./scripts/dx_tool.sh add${NC}"
    echo -e "  3. Add category to CMakeLists.txt CATEGORIES (if needed)"
}

# =============================================================================
# 8. Validate All
# =============================================================================
do_validate() {
    print_divider "Validate All Models"

    local lang_choice
    lang_choice=$(prompt_select "Validation target:" "C++" "Python" "Both")

    local video_choice
    video_choice=$(prompt_select "Validation scope:" "Image only (fast)" "Image + Video")
    local NO_VIDEO_FLAG=""
    [[ "$video_choice" == "1" ]] && NO_VIDEO_FLAG="--no-video"

    case "$lang_choice" in
        1|3)
            _info "[C++] Starting full model validation..."
            if [[ -f "$SCRIPT_DIR/validate_models.sh" ]]; then
                bash "$SCRIPT_DIR/validate_models.sh" --lang cpp $NO_VIDEO_FLAG
            else
                _warn "validate_models.sh not found"
            fi
            ;;&
        2|3)
            _info "[Python] Starting full model validation..."
            if [[ -f "$SCRIPT_DIR/validate_models.sh" ]]; then
                bash "$SCRIPT_DIR/validate_models.sh" --lang py $NO_VIDEO_FLAG
            else
                _warn "validate_models.sh not found"
            fi
            ;;
    esac
}

# =============================================================================
# 9. Run Examples
# =============================================================================
do_run() {
    local extra_args=("${@}")

    if [[ ${#extra_args[@]} -eq 0 ]]; then
        # No arguments: delegate to run_examples.sh interactive mode
        if [[ -f "$SCRIPT_DIR/run_examples.sh" ]]; then
            bash "$SCRIPT_DIR/run_examples.sh"
        else
            _warn "run_examples.sh not found"
        fi
        return
    fi

    _info "Running: scripts/run_examples.sh ${extra_args[*]}"
    if [[ -f "$SCRIPT_DIR/run_examples.sh" ]]; then
        bash "$SCRIPT_DIR/run_examples.sh" "${extra_args[@]}"
    else
        _warn "run_examples.sh not found"
    fi
}

# =============================================================================
# 10. Benchmark
# =============================================================================
do_bench() {
    local extra_args=("${@}")

    if [[ ${#extra_args[@]} -eq 0 ]]; then
        print_divider "Model Benchmark"

        local lang_choice
        lang_choice=$(prompt_select "Benchmark target:" "C++" "Python" "Both")
        case "$lang_choice" in
            1) extra_args=("--lang" "cpp") ;;
            2) extra_args=("--lang" "py") ;;
            3) extra_args=("--lang" "both") ;;
        esac

        local input_choice
        input_choice=$(prompt_select "Input type:" "Image (quick)" "Video (long-running)")
        case "$input_choice" in
            2) extra_args+=("--video") ;;
        esac

        local loops
        read -rp "$(echo -e "${CYAN}▸${NC} Iterations (Enter = 3): ")" loops
        loops="${loops:-3}"
        extra_args+=("--loops" "$loops")

        local warmup
        read -rp "$(echo -e "${CYAN}▸${NC} Warmup count (Enter = 1): ")" warmup
        warmup="${warmup:-1}"
        extra_args+=("--warmup" "$warmup")

        _prompt_filter extra_args
    fi

    _info "Running: scripts/bench_models.sh ${extra_args[*]}"
    if [[ -f "$SCRIPT_DIR/bench_models.sh" ]]; then
        bash "$SCRIPT_DIR/bench_models.sh" "${extra_args[@]}"
    else
        _warn "bench_models.sh not found"
    fi
}

# =============================================================================
# Help
# =============================================================================
do_help() {
    echo -e "${BOLD}DX Model Tool${NC}  ${DIM}v${VERSION}${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC}"
    echo "  dx_tool.sh                       Interactive menu"
    echo ""
    echo -e "${BOLD}Model Management:${NC}"
    echo "  dx_tool.sh add                   Add new model (interactive)"
    echo "  dx_tool.sh delete <MODEL>        Delete model"
    echo "  dx_tool.sh extract               Extract standalone package"
    echo ""
    echo -e "${BOLD}Query:${NC}"
    echo "  dx_tool.sh list [FILTER]         List models (category filter)"
    echo "  dx_tool.sh search <KEYWORD>      Search by model name/category"
    echo "  dx_tool.sh info <MODEL>          Model detail info"
    echo ""
    echo -e "${BOLD}Others:${NC}"
    echo "  dx_tool.sh new-task              Create new task type skeleton"
    echo "  dx_tool.sh validate              Validate all model structures"
    echo "  dx_tool.sh run [OPTS...]         Run examples (--lang cpp|py|both, etc.)"
    echo "  dx_tool.sh bench [OPTS...]       Benchmark models (--lang, --loops, --filter, etc.)"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  dx_tool.sh list det              Show object_detection only"
    echo "  dx_tool.sh search yolov8         Search yolov8-related models"
    echo "  dx_tool.sh info yolov8           YOLOv8 detail info"
    echo "  dx_tool.sh info object_detection/yolov8"
    echo "  dx_tool.sh delete yolov5_custom  Delete model"
}

# =============================================================================
# Main Entry Point
# =============================================================================
main() {
    local command="${1:-}"
    local arg="${2:-}"

    # Direct command mode
    case "$command" in
        add)            do_add_model; return ;;
        extract|export) do_extract_model_package; return ;;
        list)           do_list_models "$arg"; return ;;
        search)         do_search "$arg"; return ;;
        info)           do_info "$arg"; return ;;
        delete|rm)      do_delete_model "$arg"; return ;;
        new-task)       do_new_task; return ;;
        validate)       do_validate; return ;;
        run)            shift; do_run "$@"; return ;;
        bench)          shift; do_bench "$@"; return ;;
        help|--help|-h) do_help; return ;;
        "")
            ;; # fall through to interactive
        *)
            _err "Unknown command: $command"
            echo ""
            do_help
            return 1
            ;;
    esac

    # Interactive menu mode
    while true; do
        print_header
        echo ""
        read -rp "Select: " choice
        case "$choice" in
            1) do_add_model ;;
            2) do_extract_model_package ;;
            3) do_list_models ;;
            4) do_search ;;
            5) do_info ;;
            6) do_delete_model ;;
            7) do_new_task ;;
            8) do_validate ;;
            9) do_run ;;
            10) do_bench ;;
            0) echo -e "\n${DIM}Exiting${NC}"; exit 0 ;;
            *) _err "Please enter a number between 0 and 10" ;;
        esac
        echo ""
        read -rp "Press Enter to continue... " _
    done
}

main "$@"
