#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

BASE_URL="https://sdk.deepx.ai/"

# default value
VIDEO_VERSION="v3.2.2"
SOURCE_PATH="res/video/sample_videos_${VIDEO_VERSION}.tar.gz"
OUTPUT_DIR="$SCRIPT_DIR/assets/videos"
SYMLINK_TARGET_PATH=""
SYMLINK_ARGS=""
FORCE_ARGS="--force"
IGNORED_INTERNAL_OPTS=()

# Function to display help message
show_help() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Options:"
  echo "  [--force]                  Force overwrite if the file already exists (default)"
  echo "  [--no-force]               Skip download if the file already exists"
  echo "  [--help]                   Show this help message"

  if [ "$1" == "error" ]; then
    echo "Error: Invalid or missing arguments."
    exit 1
  fi
  exit 0
}

main() {
    SCRIPT_DIR=$(realpath "$(dirname "$0")")

    GET_RES_CMD="$SCRIPT_DIR/scripts/get_resource.sh --src_path=$SOURCE_PATH --output=$OUTPUT_DIR $SYMLINK_ARGS $FORCE_ARGS --extract"
    echo "Get Resources from remote server ..."
    echo "$GET_RES_CMD"

    $GET_RES_CMD || {
        local error_msg="Get resource failed!"
        local hint_msg="If the issue persists, please try again with sudo and the --force option, like this: 'sudo ./setup_sample_videos.sh --force'."
        local origin_cmd="" # no need to run origin command
        local suggested_action_cmd="sudo $GET_RES_CMD --force"

        # handle_cmd_failure function arguments
        #   - local error_message=$1
        #   - local hint_message=$2
        #   - local origin_cmd=$3
        #   - local suggested_action_cmd=$4
        handle_cmd_failure "$error_msg" "$hint_msg" "$origin_cmd" "$suggested_action_cmd"
    }

    # When no symlink_target_path is set, get_resource.sh extracts into a versioned subdir
    # (e.g. assets/videos/sample_videos_v3.2.2/). Flatten it so videos live directly in OUTPUT_DIR.
    # The extracted directory name may differ from the tar filename, so detect any single
    # subdirectory that contains media files and flatten it.
    if [ -z "$SYMLINK_TARGET_PATH" ]; then
        local subdirs=()
        for d in "$OUTPUT_DIR"/*/; do
            [ -d "$d" ] && [[ "$(basename "$d")" != "download" ]] && subdirs+=("$d")
        done
        if [ ${#subdirs[@]} -eq 1 ]; then
            local subdir="${subdirs[0]}"
            local media_count
            media_count=$(find "$subdir" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.mov' -o -name '*.avi' -o -name '*.mkv' \) 2>/dev/null | wc -l)
            if [ "$media_count" -gt 0 ]; then
                print_colored "Flattening $subdir → $OUTPUT_DIR" "INFO"
                shopt -s dotglob
                mv "$subdir"* "$OUTPUT_DIR/"
                shopt -u dotglob
                rmdir "$subdir" 2>/dev/null || true
                print_colored "[OK] Videos moved directly into $OUTPUT_DIR" "INFO"
            fi
        fi
    fi
}

# parse args
for i in "$@"; do
    case "$1" in
        --src_path=*)
            SOURCE_PATH="${1#*=}"
            ;;
        --output=*)
            OUTPUT_DIR="${1#*=}"

            # Symbolic link cannot be created when output_dir is the current directory.
            OUTPUT_REAL_DIR=$(readlink -f "$OUTPUT_DIR")
            CURRENT_REAL_DIR=$(readlink -f "./")
            if [ "$OUTPUT_REAL_DIR" == "$CURRENT_REAL_DIR" ]; then
                echo "'--output' is the same as the current directory. Please specify a different directory."
                exit 1
            fi
            ;;
        --symlink_target_path=*)
            SYMLINK_TARGET_PATH="${1#*=}"
            SYMLINK_ARGS="--symlink_target_path=$SYMLINK_TARGET_PATH"
            ;;
        # The --internal local-copy mode is gone. Its options still parse so
        # existing command lines (CI jobs) keep working; they are ignored.
        --internal)
            IGNORED_INTERNAL_OPTS+=("$1")
            ;;
        --internal-path=*)
            IGNORED_INTERNAL_OPTS+=("${1%%=*}")
            ;;
        --force)
            FORCE_ARGS="--force"
            ;;
        --no-force)
            FORCE_ARGS=""
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help "error"
            ;;
    esac
    shift
done

if [ ${#IGNORED_INTERNAL_OPTS[@]} -gt 0 ]; then
    print_colored "Ignoring ${IGNORED_INTERNAL_OPTS[*]}: the --internal local-copy mode was removed (the options go in a future release); sample videos come from S3." "WARNING"
fi

main

exit 0
