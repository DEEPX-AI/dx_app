#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

BASE_URL="https://sdk.deepx.ai/"

# default value
VIDEO_VERSION="v3.1.0"
SOURCE_PATH="res/video/sample_videos_${VIDEO_VERSION}.tar.gz"
OUTPUT_DIR="$SCRIPT_DIR/assets/videos"
SYMLINK_TARGET_PATH=""
SYMLINK_ARGS=""
FORCE_ARGS="--force"
INTERNAL_FLAG=""
DEFAULT_INTERNAL_VIDEO_PATH="/mnt/regression_storage/atd/sample_videos_${VIDEO_VERSION}.tar.gz"
INTERNAL_VIDEO_PATH="$DEFAULT_INTERNAL_VIDEO_PATH"

# Function to display help message
show_help() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Options:"
  echo "  [--force]                  Force overwrite if the file already exists (default)"
  echo "  [--no-force]               Skip download if the file already exists"
  echo "  [--internal]               Use local mount instead of S3 (internal/air-gapped network)"
  echo "  [--internal-path=<path>]   Local video archive for --internal mode"
  echo "                             (default: $DEFAULT_INTERNAL_VIDEO_PATH)"
  echo "  [--help]                   Show this help message"

  if [ "$1" == "error" ]; then
    echo "Error: Invalid or missing arguments."
    exit 1
  fi
  exit 0
}

main_internal() {
    local src_file="$INTERNAL_VIDEO_PATH"

    if [ ! -f "$src_file" ]; then
        print_colored "Internal video archive not found: $src_file" "ERROR"
        exit 1
    fi

    # Use the same versioned subdir name as S3 mode (get_resource.sh) so that
    # both modes produce identical directory layouts and symlinks.
    local versioned_dir="sample_videos_${VIDEO_VERSION}"

    # Determine where to extract
    local extract_base
    local extract_target
    if [ -n "$SYMLINK_TARGET_PATH" ]; then
        extract_base="$SYMLINK_TARGET_PATH"
    else
        extract_base="$OUTPUT_DIR"
    fi
    extract_target="$extract_base/$versioned_dir"

    if [ -d "$extract_target" ] && [ -z "$FORCE_ARGS" ]; then
        print_colored "Video directory already exists ($extract_target) — skipping" "INFO"
    else
        # Clean up stale artifacts from either mode to avoid leftover dirs
        rm -rf "$extract_target"
        # Remove legacy "sample_videos" dir if it exists (old internal mode name)
        if [ -n "$SYMLINK_TARGET_PATH" ] && [ -d "$extract_base/sample_videos" ]; then
            rm -rf "$extract_base/sample_videos"
        fi
        mkdir -p "$extract_target"
        print_colored "Extracting $src_file → $extract_target" "INFO"
        local first_entry
        first_entry=$(tar tf "$src_file" | head -n 1)
        if [[ "$first_entry" == */* ]]; then
            tar xf "$src_file" --strip-components=1 -C "$extract_target"
        else
            tar xf "$src_file" -C "$extract_target"
        fi
        if [ $? -ne 0 ]; then
            print_colored "Extraction failed!" "ERROR"
            exit 1
        fi
    fi

    # Create symlink OUTPUT_DIR → extract_target (mirrors get_resource.sh behaviour)
    if [ -n "$SYMLINK_TARGET_PATH" ]; then
        if [ -L "$OUTPUT_DIR" ] || [ -d "$OUTPUT_DIR" ]; then
            rm -rf "$OUTPUT_DIR"
        fi
        mkdir -p "$(dirname "$OUTPUT_DIR")"
        ln -s "$(readlink -f "$extract_target")" "$OUTPUT_DIR"
        print_colored "Created symbolic link: $OUTPUT_DIR → $(readlink -f "$extract_target")" "INFO"
    else
        # No symlink mode: flatten versioned subdir into OUTPUT_DIR (same as S3 mode)
        local media_count
        media_count=$(find "$extract_target" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.mov' -o -name '*.avi' -o -name '*.mkv' \) 2>/dev/null | wc -l)
        if [ "$media_count" -gt 0 ]; then
            print_colored "Flattening $extract_target → $OUTPUT_DIR" "INFO"
            shopt -s dotglob
            mv "$extract_target"/* "$OUTPUT_DIR/"
            shopt -u dotglob
            rmdir "$extract_target" 2>/dev/null || true
            print_colored "[OK] Videos moved directly into $OUTPUT_DIR" "INFO"
        fi
    fi

    print_colored "[OK] Sample videos setup complete (internal mode)" "INFO"
}

main() {
    SCRIPT_DIR=$(realpath "$(dirname "$0")")

    # Auto-detect internal mode if the local archive exists
    if [ -z "$INTERNAL_FLAG" ] && [ -f "$INTERNAL_VIDEO_PATH" ]; then
        print_colored "Local video archive detected ($INTERNAL_VIDEO_PATH) — switching to internal mode automatically" "INFO"
        INTERNAL_FLAG="--internal"
    fi

    if [ -n "$INTERNAL_FLAG" ]; then
        main_internal
        return
    fi

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
    # (e.g. assets/videos/sample_videos_v3.1.0/). Flatten it so videos live directly in OUTPUT_DIR.
    # The internal directory name may differ from the tar filename, so detect any single
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
        --internal)
            INTERNAL_FLAG="--internal"
            ;;
        --internal-path=*)
            INTERNAL_VIDEO_PATH="${1#*=}"
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

main

exit 0
