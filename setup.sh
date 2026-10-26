#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
RUNTIME_PATH=$(realpath -s "${SCRIPT_DIR}/..")
DX_AS_PATH=$(realpath -s "${RUNTIME_PATH}/..")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh
source ${SCRIPT_DIR}/scripts/gui_env.sh

# --- Initialize variables ---
ENABLE_DEBUG_LOGS=0   # New flag for debug logging
DOCKER_VOLUME_PATH=${DOCKER_VOLUME_PATH}
FORCE_ARGS="--force"
FORCE_REMOVE_MODELS=0
FORCE_REMOVE_VIDEOS=0
MANIFEST_OVERRIDE=""
DOWNLOAD_ALL_ARGS=""
DRY_RUN_ARG=""
LIST_ARG=""
WORKERS_ARG=""
NO_JSON_ARG=""
CATEGORY_ARG=""
DEMO_MODELS_ARG=""
MODELS_ARG=""
MODELS_ARGS=()
IGNORED_INTERNAL_OPTS=()

# If the user only asked for downloader help, forward to the inner setup_sample_models helper
if [ "$#" -eq 1 ]; then
    case "$1" in
        -h|-help)
            exec "${SCRIPT_DIR}/setup_sample_models.sh" -h
            ;;
    esac
fi

pushd $SCRIPT_DIR

# Function to display help message
show_help() {
    print_colored "Usage: $(basename "$0") [OPTIONS]" "YELLOW"
    print_colored "Options:" "GREEN"
    print_colored "  --docker_volume_path=<path>    Set Docker volume path (required in container mode)" "GREEN"
    print_colored "  [--manifest=<path>]            Use an alternate manifest JSON file for model downloads" "GREEN"
    print_colored "  [--all]                        Download all models non-interactively" "GREEN"
    print_colored "  [--dry-run]                    List models that would be downloaded without downloading" "GREEN"
    print_colored "  [--list]                       List available models without downloading" "GREEN"
    print_colored "  [--workers=<N>]                Parallel download threads (default: 4)" "GREEN"
    print_colored "  [--no-json]                    Skip JSON file downloads" "GREEN"
    print_colored "  [--category=<name>]            Download models of a specific category only" "GREEN"
    print_colored "  [--demo-models]                Download only models used by run_demo.sh/run_demo.bat" "GREEN"
    print_colored "  [--models=<m1>[,m2...]]        Download specific models by name, .dxnn file name or" "GREEN"
    print_colored "                                 registry model_name (also: --models <m1> [m2...])" "GREEN"
    print_colored "  [--force]                      Force overwrite if the file already exists (default)" "GREEN"
    print_colored "  [--no-force]                   Skip download if the file already exists" "GREEN"
    print_colored "  [--force-remove-models]        Force remove models if they exist" "GREEN"
    print_colored "  [--force-remove-videos]        Force remove videos if they exist" "GREEN"
    print_colored "  [--verbose]                    Enable verbose (debug) logging." "GREEN"
    print_colored "  [--help]                       Show this help message" "GREEN"
    print_colored "Environment:" "GREEN"
    print_colored "  DXAPP_SETUP_PYTHON=<python>    Python that runs the model downloader (default: the active" "GREEN"
    print_colored "                                 virtualenv, else ../venv-dx-runtime, else python3)" "GREEN"
    print_colored "  DXAPP_TLS_RELAX_X509_STRICT=1  Behind a TLS-inspecting proxy on Python 3.13+: turn off only" "GREEN"
    print_colored "                                 X.509 strict mode (chain and host name still verified)" "GREEN"

    if [ "$1" == "error" ] && [[ ! -n "$2" ]]; then
        print_colored "Invalid or missing arguments." "ERROR"
        exit 1
    elif [ "$1" == "error" ] && [[ -n "$2" ]]; then
        print_colored "$2" "ERROR"
        exit 1
    elif [[ "$1" == "warn" ]] && [[ -n "$2" ]]; then
        print_colored "$2" "WARNING"
        return 0
    fi
    exit 0
}

# Append model names to MODELS_ARGS, splitting on commas and whitespace.
add_model_names() {
    local _names _name
    IFS=', ' read -r -a _names <<< "$1"
    for _name in "${_names[@]}"; do
        [ -n "$_name" ] && MODELS_ARGS+=("$_name")
    done
}

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --docker_volume_path=*)
            DOCKER_VOLUME_PATH="${1#*=}"
            shift
            ;;
        --docker_volume_path)
            DOCKER_VOLUME_PATH="$2"
            shift 2
            ;;
        # The --internal local-copy mode is gone. Its options still parse so
        # existing command lines (CI jobs) keep working; they are ignored.
        --internal)
            IGNORED_INTERNAL_OPTS+=("$1")
            shift
            ;;
        --internal-path=*|--internal-video-path=*)
            IGNORED_INTERNAL_OPTS+=("${1%%=*}")
            shift
            ;;
        --internal-path|--internal-video-path)
            IGNORED_INTERNAL_OPTS+=("$1")
            shift 2
            ;;
        --all)
            DOWNLOAD_ALL_ARGS="--all"
            shift
            ;;
        --dry-run)
            DRY_RUN_ARG="--dry-run"
            shift
            ;;
        --list)
            LIST_ARG="--list"
            shift
            ;;
        --workers=*)
            WORKERS_ARG="--workers=${1#*=}"
            shift
            ;;
        --workers)
            WORKERS_ARG="--workers=$2"
            shift 2
            ;;
        --no-json)
            NO_JSON_ARG="--no-json"
            shift
            ;;
        --category=*)
            CATEGORY_ARG="--category=${1#*=}"
            shift
            ;;
        --category)
            CATEGORY_ARG="--category=$2"
            shift 2
            ;;
        --demo-models)
            DEMO_MODELS_ARG="--demo-models"
            shift
            ;;
        --models=*|--models)
            # --models=a,b | --models=a b | --models a b (names may repeat
            # the option; all of them are kept)
            _models_count=${#MODELS_ARGS[@]}
            [[ "$1" == --models=* ]] && add_model_names "${1#*=}"
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                add_model_names "$1"
                shift
            done
            if [ ${#MODELS_ARGS[@]} -eq "$_models_count" ]; then
                show_help "error" "--models requires at least one model name"
            fi
            MODELS_ARG="--models ${MODELS_ARGS[*]}"
            ;;
        --manifest=*)
            MANIFEST_OVERRIDE="${1#*=}"
            shift
            ;;
        --manifest)
            MANIFEST_OVERRIDE="$2"
            shift 2
            ;;
        --force)
            FORCE_ARGS="--force"
            shift
            ;;
        --no-force)
            # Forward the negative flag explicitly: both child layers default to
            # force-on, so omitting the flag would silently mean "--force".
            FORCE_ARGS="--no-force"
            shift
            ;;
        --force-remove-models)
            FORCE_REMOVE_MODELS=1
            shift
            ;;
        --force-remove-videos)
            FORCE_REMOVE_VIDEOS=1
            shift
            ;;
        --verbose)
            ENABLE_DEBUG_LOGS=1
            shift
            ;;
        --help|-h|-help)
            show_help
            ;;
        *)
            show_help "error" "Invalid option '$1'"
            ;;
    esac
done

if [ ${#IGNORED_INTERNAL_OPTS[@]} -gt 0 ]; then
    print_colored "Ignoring ${IGNORED_INTERNAL_OPTS[*]}: the --internal local-copy mode was removed (the options go in a future release); models and sample videos come from S3." "WARNING"
fi

print_colored "======== PATH INFO =========" "DEBUG"
print_colored "RUNTIME_PATH($RUNTIME_PATH)" "DEBUG"
print_colored "DX_AS_PATH($DX_AS_PATH)" "DEBUG"

# Default values
print_colored "=== DOCKER_VOLUME_PATH($DOCKER_VOLUME_PATH) is set ===" "INFO"

setup_assets() {
    MODEL_PATH=./assets/models
    VIDEO_PATH=./assets/videos
    CONTAINER_MODE=false

    # Check if running in a container
    if grep -qE "/docker|/lxc|/containerd" /proc/1/cgroup || [ -f /.dockerenv ]; then
        CONTAINER_MODE=true
        print_colored "(container mode detected)" "INFO"
        
        if [ -z "$DOCKER_VOLUME_PATH" ]; then
            show_help "error" "--docker_volume_path must be provided in container mode."
            exit 1
        fi

        SETUP_MODEL_ARGS="--output=${MODEL_PATH} --symlink_target_path=${DOCKER_VOLUME_PATH}/res/models"
        SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=${DOCKER_VOLUME_PATH}/res/videos"
    else
        print_colored "(host mode detected)" "INFO"
        WORKSPACE_RES="${DX_AS_PATH}/workspace/res"
        if mkdir -p "${WORKSPACE_RES}/models" "${WORKSPACE_RES}/videos" 2>/dev/null; then
            SETUP_MODEL_ARGS="--output=${MODEL_PATH} --symlink_target_path=${WORKSPACE_RES}/models"
            SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=${WORKSPACE_RES}/videos"
        else
            print_colored "shared workspace '${DX_AS_PATH}/workspace' is not writable; downloading into local ${MODEL_PATH} / ${VIDEO_PATH} instead" "WARNING"
            SETUP_MODEL_ARGS="--output=${MODEL_PATH}"
            SETUP_VIDEO_ARGS="--output=${VIDEO_PATH}"
        fi
    fi

    if [ -n "$MANIFEST_OVERRIDE" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS --manifest=${MANIFEST_OVERRIDE}"
    fi
    if [ -n "$DOWNLOAD_ALL_ARGS" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $DOWNLOAD_ALL_ARGS"
    fi
    if [ -n "$DRY_RUN_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $DRY_RUN_ARG"
    fi
    if [ -n "$LIST_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $LIST_ARG"
    fi
    if [ -n "$WORKERS_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $WORKERS_ARG"
    fi
    if [ -n "$NO_JSON_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $NO_JSON_ARG"
    fi
    if [ -n "$CATEGORY_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $CATEGORY_ARG"
    fi
    if [ -n "$DEMO_MODELS_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $DEMO_MODELS_ARG"
    fi
    if [ -n "$MODELS_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $MODELS_ARG"
    fi

    print_colored " MODEL_PATH: ${MODEL_PATH}" "INFO"
    MODEL_REAL_PATH=$(readlink -f "$MODEL_PATH")
    # Resolve force per asset kind so --force-remove-models cannot turn force
    # back on for videos.
    MODEL_FORCE_ARGS="$FORCE_ARGS"
    if [ $FORCE_REMOVE_MODELS -eq 1 ]; then
        MODEL_FORCE_ARGS="--force"
    fi
    # Always delegate to the downloader: it applies --force / --no-force per
    # file, so --no-force keeps existing models and fetches only what is
    # missing. Skipping the whole step here would leave partial asset trees
    # unrepaired.
    print_colored " Running setup models script... ($MODEL_REAL_PATH)" "INFO"
    # On failure remove only a model path this run created: an existing one
    # (a symlink, or models downloaded earlier) stays, since one failed file
    # now fails the whole step.
    MODEL_PATH_EXISTED=0
    if [ -e "$MODEL_PATH" ] || [ -L "$MODEL_PATH" ]; then
        MODEL_PATH_EXISTED=1
    fi
    ./setup_sample_models.sh $SETUP_MODEL_ARGS $MODEL_FORCE_ARGS || {
        print_colored "Setup models script failed." "ERROR"
        [ "$MODEL_PATH_EXISTED" -eq 0 ] && rm -rf "$MODEL_PATH"
        exit 1
    }

    if [ -n "$LIST_ARG" ] || [ -n "$DRY_RUN_ARG" ]; then
        print_colored "Skipping video setup for list/dry-run mode." "INFO"
    else
        print_colored "VIDEO_PATH: ${VIDEO_PATH}" "INFO"
        VIDEO_REAL_PATH=$(readlink -f "$VIDEO_PATH")
        # Resolve force per asset kind so --force-remove-videos cannot turn
        # force back on for models.
        VIDEO_FORCE_ARGS="$FORCE_ARGS"
        if [ "${FORCE_REMOVE_VIDEOS:-0}" -eq 1 ]; then
            VIDEO_FORCE_ARGS="--force"
        fi
        # Always delegate: under --no-force get_resource.sh keeps the cached
        # archive, so nothing is re-downloaded.
        print_colored " Running setup videos script... ($VIDEO_REAL_PATH)" "INFO"
        VIDEO_PATH_EXISTED=0
        if [ -e "$VIDEO_PATH" ] || [ -L "$VIDEO_PATH" ]; then
            VIDEO_PATH_EXISTED=1
        fi
        ./setup_sample_videos.sh $SETUP_VIDEO_ARGS $VIDEO_FORCE_ARGS || {
            print_colored "Setup videos script failed." "ERROR"
            [ "$VIDEO_PATH_EXISTED" -eq 0 ] && rm -rf "$VIDEO_PATH"
            exit 1
        }
    fi

    print_colored "[OK] Sample models and videos setup complete" "INFO"
}

# Python the examples run with, in the order the help text documents.
resolve_gui_python() {
    if [ -n "${DXAPP_SETUP_PYTHON}" ]; then
        echo "${DXAPP_SETUP_PYTHON}"
    elif [ -n "${VIRTUAL_ENV}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
        echo "${VIRTUAL_ENV}/bin/python3"
    elif [ -x "${RUNTIME_PATH}/venv-dx-runtime/bin/python3" ]; then
        echo "${RUNTIME_PATH}/venv-dx-runtime/bin/python3"
    else
        echo "python3"
    fi
}

main() {
    setup_assets
    if [ -z "$LIST_ARG" ] && [ -z "$DRY_RUN_ARG" ]; then
        dxapp_link_cv2_qt_fonts "$(resolve_gui_python)"
    fi
}

main

popd
