#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
RUNTIME_PATH=$(realpath -s "${SCRIPT_DIR}/..")
DX_AS_PATH=$(realpath -s "${RUNTIME_PATH}/..")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

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
MODELS_ARG=""
INTERNAL_ARG=""
INTERNAL_PATH_ARG=""

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
    print_colored "  [--models=<m1> [m2...]]        Download specific models by name" "GREEN"
    print_colored "  [--force]                      Force overwrite if the file already exists (default)" "GREEN"
    print_colored "  [--no-force]                   Skip download if the file already exists" "GREEN"
    print_colored "  [--force-remove-models]        Force remove models if they exist" "GREEN"
    print_colored "  [--force-remove-videos]        Force remove videos if they exist" "GREEN"
    print_colored "  [--internal]                   Use local mount instead of S3 (internal/air-gapped network)" "GREEN"
    print_colored "  [--internal-path=<path>]       Local model directory for --internal mode" "GREEN"
    print_colored "                                 (default: /mnt/regression_storage/atd/models_v3.1.0)" "GREEN"
    print_colored "  [--verbose]                    Enable verbose (debug) logging." "GREEN"
    print_colored "  [--help]                       Show this help message" "GREEN"

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
        --internal)
            INTERNAL_ARG="--internal"
            shift
            ;;
        --internal-path=*)
            INTERNAL_PATH_ARG="--internal-path=${1#*=}"
            shift
            ;;
        --internal-path)
            INTERNAL_PATH_ARG="--internal-path=$2"
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
        --models)
            shift
            MODELS_ARGS=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                MODELS_ARGS+=("$1")
                shift
            done
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
            FORCE_ARGS=""
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
        SETUP_MODEL_ARGS="--output=${MODEL_PATH} --symlink_target_path=${DX_AS_PATH}/workspace/res/models"
        SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=${DX_AS_PATH}/workspace/res/videos"
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
    if [ -n "$MODELS_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $MODELS_ARG"
    fi
    if [ -n "$INTERNAL_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $INTERNAL_ARG"
    fi
    if [ -n "$INTERNAL_PATH_ARG" ]; then
        SETUP_MODEL_ARGS="$SETUP_MODEL_ARGS $INTERNAL_PATH_ARG"
    fi
    if [ -n "$INTERNAL_ARG" ]; then
        SETUP_VIDEO_ARGS="$SETUP_VIDEO_ARGS $INTERNAL_ARG"
    fi
    if [ -n "$INTERNAL_PATH_ARG" ]; then
        SETUP_VIDEO_ARGS="$SETUP_VIDEO_ARGS $INTERNAL_PATH_ARG"
    fi

    print_colored " MODEL_PATH: ${MODEL_PATH}" "INFO"
    MODEL_REAL_PATH=$(readlink -f "$MODEL_PATH")
    # Check and set up models
    # Interactive mode always runs (user selects models; downloader skips already-existing files)
    # Non-interactive (--all) mode skips if directory already exists unless forced
    if [ ! -d "$MODEL_REAL_PATH" ] || [ "$FORCE_ARGS" != "" ] || [ $FORCE_REMOVE_MODELS -eq 1 ] || [ -z "$DOWNLOAD_ALL_ARGS" ]; then
        if [ $FORCE_REMOVE_MODELS -eq 1 ]; then
            FORCE_ARGS="--force"
        fi
        print_colored " Running setup models script... ($MODEL_REAL_PATH)" "INFO"
        ./setup_sample_models.sh $SETUP_MODEL_ARGS $FORCE_ARGS || { print_colored "Setup models script failed." "ERROR"; rm -rf $MODEL_PATH; exit 1; }
    else
        print_colored " models directory found. ($MODEL_REAL_PATH)" "INFO"
    fi

    print_colored "VIDEO_PATH: ${VIDEO_PATH}" "INFO"
    VIDEO_REAL_PATH=$(readlink -f "$VIDEO_PATH")
    # Check and set up models
    if [ ! -d "$VIDEO_REAL_PATH" ] || [ "$FORCE_ARGS" != "" ] || [ "${FORCE_REMOVE_VIDEOS:-0}" -eq 1 ]; then
        if [ "${FORCE_REMOVE_VIDEOS:-0}" -eq 1 ]; then
            FORCE_ARGS="--force"
        fi
        print_colored " Video directory not found. Running setup models script... ($VIDEO_REAL_PATH)" "INFO"
        ./setup_sample_videos.sh $SETUP_VIDEO_ARGS $FORCE_ARGS || { print_colored "Setup videos script failed." "ERROR"; rm -rf $VIDEO_PATH; exit 1; }
    else
        print_colored " Video directory found. ($VIDEO_REAL_PATH)" "INFO"
    fi

    print_colored "[OK] Sample models and videos setup complete" "INFO"
}

main() {
    setup_assets
}

main

popd
