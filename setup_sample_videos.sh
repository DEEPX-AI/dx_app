#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

BASE_URL="https://sdk.deepx.ai/"

# default value
SOURCE_PATH="res/video/sample_videos.tar.gz"
OUTPUT_DIR="$SCRIPT_DIR/assets/videos"
SYMLINK_TARGET_PATH=""
SYMLINK_ARGS=""
FORCE_ARGS=""

# Function to display help message
show_help() {
  
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Options:"
  echo "  [--force]                  Force overwrite if the file already exists"
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
        local hint_msg="If the issue persists, please try again with sudo and the --force option, like this: 'sudo ./setup_sample_models.sh --force'."
        local origin_cmd="" # no need to run origin command
        local suggested_action_cmd="sudo $GET_RES_CMD --force"

        # handle_cmd_failure function arguments
        #   - local error_message=$1
        #   - local hint_message=$2
        #   - local origin_cmd=$3
        #   - local suggested_action_cmd=$4
        handle_cmd_failure "$error_msg" "$hint_msg" "$origin_cmd" "$suggested_action_cmd"
    }
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
        --force)
            FORCE_ARGS="--force"
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

main

exit 0
