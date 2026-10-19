#!/bin/bash
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
#
# Smoke the multi-model graph on the shipped sample scenarios.
#   graph  bin/multi_model_graph_{sync,async}
#
#   ./run_multi_model_demo.sh                         # interactive menu
#   ./run_multi_model_demo.sh --list
#   ./run_multi_model_demo.sh --check                 # every sample graph, no NPU
#   ./run_multi_model_demo.sh --all                   # --check, then each scenario
#   ./run_multi_model_demo.sh --scenario hand_cascade
#   ./run_multi_model_demo.sh --scenario 1 --async

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
# shellcheck source=scripts/color_env.sh
source "$SCRIPT_DIR/scripts/color_env.sh"
# shellcheck source=scripts/common_util.sh
source "$SCRIPT_DIR/scripts/common_util.sh"

DX_APP_PATH="$SCRIPT_DIR"
GRAPH_DIR="$DX_APP_PATH/src/cpp_example/multi_model_graph"
MODEL_DIR="$DX_APP_PATH/assets/models"
OUT_DIR="$DX_APP_PATH/reports/dev/multi_model_demo"

# Scenario id, label, graph JSON, sample image.
SCENARIO_IDS=(hand_cascade logistics_volume dms)
SCENARIO_LABELS=(
    "Hand cascade (palm -> landmarks)"
    "Logistics volume (det + seg + depth)"
    "DMS (face + head pose)"
)
SCENARIO_GRAPH=(
    "$GRAPH_DIR/hand_cascade.json"
    "$GRAPH_DIR/logistics_volume.json"
    "$GRAPH_DIR/dms_headpose.json"
)
SCENARIO_IMAGE=(
    "$DX_APP_PATH/sample/img/sample_hand.jpg"
    "$DX_APP_PATH/sample/img/sample_baggage_cart_airport.jpg"
    "$DX_APP_PATH/sample/img/sample_face.jpg"
)
# In-cabin driver frame. Not shipped with the repo; used for DMS when present.
DMS_DRIVER_IMAGE="$DX_APP_PATH/sample/img/sample_driver_monitoring.png"
if [ -f "$DMS_DRIVER_IMAGE" ]; then
    SCENARIO_IMAGE[2]="$DMS_DRIVER_IMAGE"
fi

SCENARIO=""
USE_ASYNC=0
DO_CHECK=0
DO_ALL=0
DO_LIST=0
FAIL_COUNT=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Check the shipped multi-model graphs. Each scenario runs one sample image
through bin/multi_model_graph_sync (or _async with --async).

Scenarios:
  0  hand_cascade       palm -> landmarks
  1  logistics_volume   det + seg + depth
  2  dms                face + head pose

Options:
  --scenario ID|NUM   Scenario id or menu index (0-$(( ${#SCENARIO_IDS[@]} - 1 )))
  --async             Use multi_model_graph_async
  --check             Validate every sample graph (--check). Does not load the NPU.
  --all               --check, then each scenario with multi_model_graph_sync
  --list              Print scenarios and sample graphs, then exit
  --output-dir DIR    Reports and saved frames (default: reports/dev/multi_model_demo)
  --models-dir DIR    .dxnn directory (default: assets/models)
  -h, --help          Show this help

No arguments opens the menu. A non-interactive shell with no arguments prints
this help and exits.
EOF
}

note() { print_colored "$1" "INFO"; }
ok() { print_colored "$1" "OK"; }
fail() { print_colored "$1" "FAIL"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

require_file() {
    local path="$1"
    local hint="$2"
    if [ ! -f "$path" ]; then
        fail "Missing $path"
        print_colored "  $hint" "HINT"
        return 1
    fi
    return 0
}

scenario_index() {
    local key="$1"
    local i
    if [[ "$key" =~ ^[0-9]+$ ]] && [ "$key" -ge 0 ] && [ "$key" -lt ${#SCENARIO_IDS[@]} ]; then
        echo "$key"
        return 0
    fi
    for i in "${!SCENARIO_IDS[@]}"; do
        if [ "${SCENARIO_IDS[$i]}" = "$key" ]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

print_list() {
    local i graph
    echo "Scenarios:"
    for i in "${!SCENARIO_IDS[@]}"; do
        printf "  %d  %-18s %s\n" "$i" "${SCENARIO_IDS[$i]}" "${SCENARIO_LABELS[$i]}"
    done
    echo "Sample graphs:"
    shopt -s nullglob
    for graph in "$GRAPH_DIR"/*.json; do
        printf "  %s\n" "$(basename "$graph")"
    done
    shopt -u nullglob
}

ensure_runtime_path() {
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$DX_APP_PATH/lib:"*) ;;
        *) export LD_LIBRARY_PATH="$DX_APP_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}

graph_binary() {
    if [ "$USE_ASYNC" -eq 1 ]; then
        echo "$DX_APP_PATH/bin/multi_model_graph_async"
    else
        echo "$DX_APP_PATH/bin/multi_model_graph_sync"
    fi
}

run_graph() {
    local index="$1"
    local graph="${SCENARIO_GRAPH[$index]}"
    local name="${SCENARIO_IDS[$index]}"
    local image="${SCENARIO_IMAGE[$index]}"
    local mode="sync"
    local bin report output
    local -a input_args=()
    [ "$USE_ASYNC" -eq 1 ] && mode="async"
    bin="$(graph_binary)"
    report="$OUT_DIR/${name}_graph_${mode}.json"
    output="$OUT_DIR/${name}_graph_${mode}.jpg"
    require_file "$bin" "Build it with: ./build.sh --target multi_model_graph_sync multi_model_graph_async" || return 1
    require_file "$graph" "Sample graph JSON is missing" || return 1
    [ -f "$image" ] && input_args=(--input "$image")
    mkdir -p "$OUT_DIR"
    note "graph     $name  ($mode)  $(basename "$image")"
    if "$bin" \
            --graph "$graph" \
            "${input_args[@]}" \
            --model-dir "$MODEL_DIR" \
            --frames 1 \
            --report "$report" \
            --output "$output"; then
        ok "graph     $name  ($mode)  -> $report"
        return 0
    fi
    fail "graph     $name  ($mode)"
    return 1
}

check_all_graphs() {
    local graph bin status=0 count=0
    bin="$DX_APP_PATH/bin/multi_model_graph_sync"
    require_file "$bin" "Build it with: ./build.sh --target multi_model_graph_sync" || return 1
    shopt -s nullglob
    local graphs=( "$GRAPH_DIR"/*.json )
    shopt -u nullglob
    if [ ${#graphs[@]} -eq 0 ]; then
        fail "No sample graphs in $GRAPH_DIR"
        return 1
    fi
    note "Checking ${#graphs[@]} sample graphs (no NPU load)"
    for graph in "${graphs[@]}"; do
        count=$((count + 1))
        if "$bin" --check "$graph"; then
            ok "check  $(basename "$graph")"
        else
            fail "check  $(basename "$graph")"
            status=1
        fi
    done
    return "$status"
}

choose_menu() {
    local choice
    print_list
    echo
    echo "Scenario [0-$(( ${#SCENARIO_IDS[@]} - 1 )), default 0]:"
    read -r choice
    SCENARIO="${choice:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --scenario)
            [ $# -ge 2 ] || { print_colored "--scenario needs a value" "ERROR"; exit 1; }
            SCENARIO="$2"
            shift 2
            ;;
        --async) USE_ASYNC=1; shift ;;
        --check) DO_CHECK=1; shift ;;
        --all) DO_ALL=1; shift ;;
        --list) DO_LIST=1; shift ;;
        --output-dir)
            [ $# -ge 2 ] || { print_colored "--output-dir needs a value" "ERROR"; exit 1; }
            OUT_DIR="$2"
            shift 2
            ;;
        --models-dir)
            [ $# -ge 2 ] || { print_colored "--models-dir needs a value" "ERROR"; exit 1; }
            MODEL_DIR="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *)
            print_colored "Unknown option: $1" "ERROR"
            usage
            exit 1
            ;;
    esac
done

if [ "$DO_LIST" -eq 1 ]; then
    print_list
    exit 0
fi

if [ -z "$SCENARIO" ] && [ "$DO_CHECK" -eq 0 ] && [ "$DO_ALL" -eq 0 ]; then
    if [ ! -t 0 ]; then
        usage
        exit 2
    fi
    choose_menu
fi

ensure_runtime_path
cd "$DX_APP_PATH" || exit 1

if [ "$DO_CHECK" -eq 1 ] || [ "$DO_ALL" -eq 1 ]; then
    check_all_graphs || true
fi

if [ "$DO_ALL" -eq 1 ]; then
    saved_async="$USE_ASYNC"
    USE_ASYNC=0
    for i in "${!SCENARIO_IDS[@]}"; do
        run_graph "$i" || true
    done
    USE_ASYNC="$saved_async"
elif [ -n "$SCENARIO" ]; then
    if ! index="$(scenario_index "$SCENARIO")"; then
        print_colored "Unknown scenario '$SCENARIO'. Use --list." "ERROR"
        exit 1
    fi
    run_graph "$index"
fi

echo
if [ "$FAIL_COUNT" -eq 0 ]; then
    ok "Multi-model demo finished with no failures"
    exit 0
fi
print_colored "$FAIL_COUNT step(s) failed" "ERROR"
exit 1
