#!/bin/bash
#
# add_model.sh - Copy-from-reference example generator (C++/Python integrated)
#
# Creates src/{cpp,python}_example/<task>/<family>/<variant>/ by copying an
# existing variant of the same postprocessor family and renaming what names the
# variant (the C++ factory is scoped in dxapp::v_<variant>, config.json names the
# variant, the registry gains a row). The generation itself is
# scripts/add_model_variant.py; this wrapper adds --verify, --git-push and --auto-add.
#
# Usage:
#   ./add_model.sh <variant> <task> [options]
#
#   <variant>  the .dxnn stem, e.g. yolov8-n_640x640 (also the directory name)
#   <task>     the task directory: object_detection, image_classification,
#              semantic_segmentation, instance_segmentation, pose_estimation,
#              face_detection, oriented_object_detection, depth_estimation, ...
#              Legacy names (detection, classification, pose, obb, ppu, embedding,
#              image_enhancement, attribute_recognition, reid) are mapped.
#
# Options:
#   --family <name>         Family directory (default: the registry row's, else the
#                           reference's family)
#   --base-model <variant>  Copy from this existing variant
#   --postprocessor <key>   Pick the reference by registry postprocessor key
#                           (yolov5, yolov8, yolox, scrfd, efficientnet, deeplabv3, ...)
#   --lang <cpp|py|both>    Language (default: both)
#   --sync-only             Generate only sync (no async)
#   --input-size <WxH>      Model input size (default: registry row, else reference)
#   --dxnn <file>           Model file name (default: <variant>.dxnn)
#   --no-register           Do not add a registry row / test_models.conf line
#   --verify                Build and run inference verification after generation
#   --model <path>          Model .dxnn file for --verify
#   --video <path>          Video file for --verify
#   --image <path>          Image file for --verify (default: sample/img/sample_kitchen.jpg)
#   --no-video              Skip video verification when used with --verify (image only)
#   --git-push              After successful --verify, git add/commit/push (asks confirmation)
#   --yes                   Skip confirmation prompt for --git-push (auto-yes)
#   --auto-add              Generate the example of every registry row that has none
#   --category <name>       Ignored (kept for older callers; the task names the directory)
#
# Examples:
#   ./add_model.sh yolov8-n_320x320 object_detection --postprocessor yolov8
#   ./add_model.sh yolov7-custom_640x640 object_detection --base-model yolov7_640x640
#   ./add_model.sh my-det_640x640 object_detection --family my_det --postprocessor yolov5 --lang cpp
#   ./add_model.sh yolox-custom_640x640 object_detection --postprocessor yolox --verify --model model.dxnn
#   ./add_model.sh resnet-custom_224x224 image_classification --postprocessor efficientnet

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DX_APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_SRC_DIR="$DX_APP_ROOT/src/cpp_example"
PY_SRC_DIR="$DX_APP_ROOT/src/python_example"
GENERATOR="$SCRIPT_DIR/add_model_variant.py"
PYTHON_BIN="${DXAPP_SETUP_PYTHON:-python3}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[DXAPP] [INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warn()    { echo -e "${YELLOW}[DXAPP] [WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[DXAPP] [ERROR]${NC} $1"; }

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^#//;s/^ //'
    exit "${1:-0}"
}

# ============================================================================
# Parse arguments
# ============================================================================
MODEL_NAME=""
TASK_TYPE=""
LANG_MODE="both"
FAMILY=""
SYNC_ONLY=false
POSTPROCESSOR=""
BASE_MODEL=""
INPUT_SIZE=""
DXNN_FILE=""
NO_REGISTER=false
VERIFY=false
VERIFY_MODEL=""
VERIFY_VIDEO=""
VERIFY_IMAGE=""
NO_VIDEO=false
GIT_PUSH=false
AUTO_YES=false
AUTO_ADD=false

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage 0
fi

if [ $# -ge 1 ] && [ "$1" = "--auto-add" ]; then
    AUTO_ADD=true
    shift
    # An older manifest argument is accepted and ignored: the registry is the source.
    if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
        print_warn "--auto-add no longer reads a manifest; ignoring '$1'"
        shift
    fi
elif [ $# -lt 2 ]; then
    usage 1
else
    MODEL_NAME="$1"
    TASK_TYPE="$2"
    shift 2
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --lang)           LANG_MODE="$2";      shift 2 ;;
        --family)         FAMILY="$2";         shift 2 ;;
        --category)       print_warn "--category is ignored: the task names the directory"; shift 2 ;;
        --sync-only)      SYNC_ONLY=true;      shift ;;
        --postprocessor)  POSTPROCESSOR="$2";  shift 2 ;;
        --base-model)     BASE_MODEL="$2";     shift 2 ;;
        --input-size)     INPUT_SIZE="$2";     shift 2 ;;
        --dxnn)           DXNN_FILE="$2";      shift 2 ;;
        --no-register)    NO_REGISTER=true;    shift ;;
        --verify)         VERIFY=true;         shift ;;
        --model)          VERIFY_MODEL="$2";   shift 2 ;;
        --video)          VERIFY_VIDEO="$2";   shift 2 ;;
        --image)          VERIFY_IMAGE="$2";   shift 2 ;;
        --no-video)       NO_VIDEO=true;       shift ;;
        --git-push)       GIT_PUSH=true;       shift ;;
        --yes)            AUTO_YES=true;       shift ;;
        --auto-add)       AUTO_ADD=true;       shift ;;
        *)                print_error "Unknown option: $1"; usage 1 ;;
    esac
done

case "$LANG_MODE" in
    cpp)        LANGS=("cpp") ;;
    py|python)  LANGS=("py"); LANG_MODE="py" ;;
    both)       LANGS=("cpp" "py") ;;
    *)          print_error "Invalid --lang: $LANG_MODE (cpp|py|both)"; exit 1 ;;
esac

if [ ! -f "$GENERATOR" ]; then
    print_error "Generator not found: $GENERATOR"
    exit 1
fi

# ============================================================================
# generate_variant <variant> <task> [extra generator args...]
#   Runs the generator and records CPP_MODEL_DIR / PY_MODEL_DIR / VARIANT /
#   TASK_DIR / FAMILY_DIR from its summary. Returns the generator's exit code
#   (3 = the variant already exists).
# ============================================================================
CPP_MODEL_DIR=""
PY_MODEL_DIR=""
VARIANT=""
TASK_DIR=""
FAMILY_DIR=""

generate_variant() {
    local variant="$1" task="$2"
    shift 2
    local args=("$variant" "$task" --root "$DX_APP_ROOT" --lang "$LANG_MODE")
    [ "$SYNC_ONLY" = true ]     && args+=(--sync-only)
    [ "$NO_REGISTER" = true ]   && args+=(--no-register)
    [ -n "$FAMILY" ]            && args+=(--family "$FAMILY")
    [ -n "$BASE_MODEL" ]        && args+=(--base-model "$BASE_MODEL")
    [ -n "$POSTPROCESSOR" ]     && args+=(--postprocessor "$POSTPROCESSOR")
    [ -n "$INPUT_SIZE" ]        && args+=(--input-size "$INPUT_SIZE")
    [ -n "$DXNN_FILE" ]         && args+=(--dxnn "$DXNN_FILE")
    args+=("$@")

    local out rc=0
    out=$("$PYTHON_BIN" "$GENERATOR" "${args[@]}" 2>&1) || rc=$?
    echo "$out" | grep -v -E '^(CPP|PY)_DIR=|^VARIANT=|^TASK=|^FAMILY=' || true
    CPP_MODEL_DIR=$(echo "$out" | sed -n 's/^CPP_DIR=//p' | tail -1)
    PY_MODEL_DIR=$(echo "$out" | sed -n 's/^PY_DIR=//p' | tail -1)
    VARIANT=$(echo "$out" | sed -n 's/^VARIANT=//p' | tail -1)
    TASK_DIR=$(echo "$out" | sed -n 's/^TASK=//p' | tail -1)
    FAMILY_DIR=$(echo "$out" | sed -n 's/^FAMILY=//p' | tail -1)
    return $rc
}

# locate_existing <name>: fill the *_MODEL_DIR globals for a variant already in the
# tree. <name> may be the registry model_name (validate_models.sh passes it).
locate_existing() {
    local name="$1" variant d
    variant=$("$PYTHON_BIN" - "$DX_APP_ROOT" "$name" <<'PYEOF'
import json, sys
from pathlib import Path
root, name = Path(sys.argv[1]), sys.argv[2]
reg = root / "config" / "model_registry.json"
rows = json.loads(reg.read_text(encoding="utf-8")) if reg.is_file() else []
for r in rows:
    if not r.get("alias_of") and name in (r.get("variant"), r.get("model_name")):
        print(r["variant"]); break
else:
    print(name)
PYEOF
    )
    VARIANT="$variant"
    for pair in "cpp:$CPP_SRC_DIR" "py:$PY_SRC_DIR"; do
        d=$(find "${pair#*:}" -mindepth 3 -maxdepth 3 -type d -name "$variant" -not -path "*/common/*" 2>/dev/null | head -1)
        if [ -n "$d" ]; then
            TASK_DIR=$(basename "$(dirname "$(dirname "$d")")")
            FAMILY_DIR=$(basename "$(dirname "$d")")
            case "${pair%%:*}" in
                cpp) CPP_MODEL_DIR="$d" ;;
                py)  PY_MODEL_DIR="$d" ;;
            esac
        fi
    done
    return 0
}

# ============================================================================
# --auto-add: generate the example of every registry row that has none
# ============================================================================
if [ "$AUTO_ADD" = true ]; then
    print_info "Auto-add: registry rows without a ${LANG_MODE} example"
    MISSING=$("$PYTHON_BIN" - "$DX_APP_ROOT" "$LANG_MODE" <<'PYEOF'
import json, sys
from pathlib import Path
root, lang = Path(sys.argv[1]), sys.argv[2]
langs = {"cpp": ["cpp"], "py": ["py"], "both": ["cpp", "py"]}[lang]
rows = json.loads((root / "config" / "model_registry.json").read_text(encoding="utf-8"))
for r in rows:
    if r.get("alias_of") or not r.get("variant"):
        continue
    missing = []
    for l in langs:
        src = root / "src" / ("cpp_example" if l == "cpp" else "python_example")
        suffix = "_factory.hpp" if l == "cpp" else "_factory.py"
        if not (src / r["task"] / r["family"] / r["variant"] / "factory" / (r["variant"] + suffix)).is_file():
            missing.append(l)
    if missing:
        print("\t".join([r["variant"], r["task"], r["family"], ",".join(missing)]))
PYEOF
    )
    if [ -z "$MISSING" ]; then
        print_info "Nothing to do: every registry row has its example."
        exit 0
    fi
    total=$(echo "$MISSING" | grep -c .)
    print_info "Rows to process: ${total}"
    done_n=0; fail_n=0
    while IFS=$'\t' read -r variant task family langs_missing; do
        [ -z "$variant" ] && continue
        echo ""
        echo "──────────────────────────────────────────────────"
        print_info "Processing: ${task}/${family}/${variant} (missing: ${langs_missing})"
        LANG_MODE_SAVE="$LANG_MODE"
        case "$langs_missing" in
            cpp)    LANG_MODE="cpp" ;;
            py)     LANG_MODE="py" ;;
            *)      LANG_MODE="both" ;;
        esac
        if FAMILY="$family" generate_variant "$variant" "$task"; then
            done_n=$((done_n + 1))
        else
            print_error "Failed: ${variant}"
            fail_n=$((fail_n + 1))
        fi
        LANG_MODE="$LANG_MODE_SAVE"
    done <<< "$MISSING"
    echo ""
    echo "══════════════════════════════════════════════════"
    print_success "Auto-add complete: ${done_n} generated, ${fail_n} failed"
    echo "══════════════════════════════════════════════════"
    [ "$fail_n" -eq 0 ]
    exit $?
fi

# ============================================================================
# Generate the package
# ============================================================================
echo ""
echo "=============================================="
echo "  Creating model: ${MODEL_NAME} (lang: ${LANG_MODE})"
echo "=============================================="

GEN_RC=0
generate_variant "$MODEL_NAME" "$TASK_TYPE" || GEN_RC=$?
if [ "$GEN_RC" -eq 3 ]; then
    if [ "$VERIFY" = true ]; then
        print_warn "Package already exists — verifying the existing example"
        locate_existing "$MODEL_NAME"
    else
        print_error "Package already exists (see above). Nothing was written."
        exit 3
    fi
elif [ "$GEN_RC" -ne 0 ]; then
    print_error "Generation failed (exit ${GEN_RC})"
    exit "$GEN_RC"
else
    echo ""
    print_success "Package created: ${TASK_DIR}/${FAMILY_DIR}/${VARIANT}"
    for lang in "${LANGS[@]}"; do
        case "$lang" in
            cpp) [ -n "$CPP_MODEL_DIR" ] && echo -e "  ${CYAN}→ ${CPP_MODEL_DIR#"$DX_APP_ROOT/"}${NC}" ;;
            py)  [ -n "$PY_MODEL_DIR" ]  && echo -e "  ${CYAN}→ ${PY_MODEL_DIR#"$DX_APP_ROOT/"}${NC}" ;;
        esac
    done
    # The per-variant namespace rule, checked right away rather than at the next CI run.
    if [ -n "$CPP_MODEL_DIR" ] && [ -f "$SCRIPT_DIR/generate_cpp_family_layout.py" ]; then
        if "$PYTHON_BIN" "$SCRIPT_DIR/generate_cpp_family_layout.py" --variant-scope "$CPP_SRC_DIR" --check > /dev/null 2>&1; then
            print_info "variant-scope --check: OK"
        else
            print_warn "variant-scope --check reports problems; run: python3 scripts/generate_cpp_family_layout.py --variant-scope src/cpp_example --check"
        fi
    fi
fi

[ "$VERIFY" = true ] || exit 0

# ============================================================================
# --verify: Build + Run inference
# ============================================================================
set +e
echo ""
echo "========================================"
echo "  AUTO-VERIFY: ${VARIANT} (lang: ${LANG_MODE})"
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
    VERIFY_MODEL="${DX_APP_ROOT}/assets/models/${VARIANT}.dxnn"
    print_info "No --model given; using ${VERIFY_MODEL#"$DX_APP_ROOT/"}"
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
if [ ! -f "$VERIFY_MODEL" ]; then
    print_error "Model file not found: $VERIFY_MODEL"
    exit 1
fi

DISPLAY_PREFIX=""
if ! xdpyinfo >/dev/null 2>&1; then
    if command -v xvfb-run >/dev/null 2>&1; then
        DISPLAY_PREFIX="xvfb-run --auto-servernum"
    fi
fi

for lang in "${LANGS[@]}"; do
    # ================================================================
    # C++ Verify: Build (build.sh --target) + Run (bin/<variant>_{sync,async})
    # ================================================================
    if [ "$lang" = "cpp" ]; then
        echo ""
        echo -e "${BLUE}━━━ C++ Build ━━━${NC}"
        SYNC_TARGET="${VARIANT}_sync"
        ASYNC_TARGET="${VARIANT}_async"
        BUILD_TARGETS=("$SYNC_TARGET")
        if [ "$SYNC_ONLY" = false ] && [ -f "${CPP_MODEL_DIR}/${ASYNC_TARGET}.cpp" ]; then
            BUILD_TARGETS+=("$ASYNC_TARGET")
        fi
        run_test "cpp_build" "cd '$DX_APP_ROOT' && ./build.sh --target ${BUILD_TARGETS[*]} 2>&1"

        SYNC_BIN="${DX_APP_ROOT}/bin/${SYNC_TARGET}"
        ASYNC_BIN="${DX_APP_ROOT}/bin/${ASYNC_TARGET}"
        [ -x "$SYNC_BIN" ]  || SYNC_BIN=$(find "$DX_APP_ROOT"/build_* -name "$SYNC_TARGET" -type f -executable 2>/dev/null | head -1)
        [ -x "$ASYNC_BIN" ] || ASYNC_BIN=$(find "$DX_APP_ROOT"/build_* -name "$ASYNC_TARGET" -type f -executable 2>/dev/null | head -1)
        if [ -z "$DISPLAY_PREFIX" ] && ! xdpyinfo >/dev/null 2>&1; then
            print_warn "No display and no xvfb-run; C++ runs use --no-display"
            CPP_DISPLAY_ARGS="--no-display"
        else
            CPP_DISPLAY_ARGS=""
        fi

        echo ""
        echo -e "${BLUE}━━━ C++ Inference ━━━${NC}"
        if [ -n "$SYNC_BIN" ] && [ -x "$SYNC_BIN" ]; then
            run_test "cpp_sync_image" \
                "cd '$DX_APP_ROOT' && $DISPLAY_PREFIX '$SYNC_BIN' -m '$VERIFY_MODEL' -i '$VERIFY_IMAGE' $CPP_DISPLAY_ARGS 2>&1"
            if [ -n "$VERIFY_VIDEO" ]; then
                run_test "cpp_sync_video" \
                    "cd '$DX_APP_ROOT' && $DISPLAY_PREFIX '$SYNC_BIN' -m '$VERIFY_MODEL' -v '$VERIFY_VIDEO' $CPP_DISPLAY_ARGS 2>&1"
            fi
        else
            echo -e "${YELLOW}[SKIP]${NC} C++ sync binary not found: ${SYNC_TARGET}"
            SKIP=$((SKIP + 1))
        fi
        if [ -n "$ASYNC_BIN" ] && [ -x "$ASYNC_BIN" ] && [ -n "$VERIFY_VIDEO" ]; then
            run_test "cpp_async_video" \
                "cd '$DX_APP_ROOT' && $DISPLAY_PREFIX '$ASYNC_BIN' -m '$VERIFY_MODEL' -v '$VERIFY_VIDEO' $CPP_DISPLAY_ARGS 2>&1"
        fi
    fi

    # ================================================================
    # Python Verify: Run the variant's entry scripts
    # ================================================================
    if [ "$lang" = "py" ]; then
        PYTHON_CMD="python3"
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

        PY_SYNC_SCRIPT="${PY_MODEL_DIR}/${VARIANT}_sync.py"
        PY_ASYNC_SCRIPT="${PY_MODEL_DIR}/${VARIANT}_async.py"
        PY_SYNC_CPP_PP_SCRIPT="${PY_MODEL_DIR}/${VARIANT}_sync_cpp_postprocess.py"

        PY_DISPLAY_ARGS=""
        if ! xdpyinfo >/dev/null 2>&1; then
            PY_DISPLAY_ARGS="--no-display"
        fi

        echo ""
        echo -e "${BLUE}━━━ Python Inference ━━━${NC}"
        if [ ! -f "$PY_SYNC_SCRIPT" ]; then
            print_error "Python sync entry not found: ${PY_SYNC_SCRIPT}"
            FAIL=$((FAIL + 1))
        else
            run_test "py_sync_image" \
                "cd '$DX_APP_ROOT' && $PYTHON_CMD '$PY_SYNC_SCRIPT' -m '$VERIFY_MODEL' -i '$VERIFY_IMAGE' $PY_DISPLAY_ARGS 2>&1"
            if [ -n "$VERIFY_VIDEO" ]; then
                run_test "py_sync_video" \
                    "cd '$DX_APP_ROOT' && $PYTHON_CMD '$PY_SYNC_SCRIPT' -m '$VERIFY_MODEL' -v '$VERIFY_VIDEO' $PY_DISPLAY_ARGS 2>&1"
            fi
        fi
        if [ "$SYNC_ONLY" = false ] && [ -f "$PY_ASYNC_SCRIPT" ] && [ -n "$VERIFY_VIDEO" ]; then
            run_test "py_async_video" \
                "cd '$DX_APP_ROOT' && $PYTHON_CMD '$PY_ASYNC_SCRIPT' -m '$VERIFY_MODEL' -v '$VERIFY_VIDEO' $PY_DISPLAY_ARGS 2>&1"
        fi
        if [ -f "$PY_SYNC_CPP_PP_SCRIPT" ]; then
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

    COMMIT_MSG="feat: add model ${VARIANT} (${TASK_DIR}/${FAMILY_DIR}, ${LANG_MODE}) - verified"

    echo ""
    print_info "Remote : ${GIT_REMOTE}"
    print_info "Branch : ${GIT_BRANCH}"
    print_info "Commit : ${COMMIT_MSG}"
    echo ""

    [ -n "$CPP_MODEL_DIR" ] && git add "$CPP_MODEL_DIR"
    [ -n "$PY_MODEL_DIR" ]  && git add "$PY_MODEL_DIR"
    [ "$NO_REGISTER" = true ] || git add "$DX_APP_ROOT/config/model_registry.json" "$DX_APP_ROOT/config/test_models.conf"

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
