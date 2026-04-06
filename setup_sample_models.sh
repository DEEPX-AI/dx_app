#!/usr/bin/env bash
# Thin wrapper: delegate all work to scripts/download_models.py
SCRIPT_DIR=$(realpath "$(dirname "$0")")
source ${SCRIPT_DIR}/scripts/color_env.sh || true
source ${SCRIPT_DIR}/scripts/common_util.sh || true

DOWNLOADER="${SCRIPT_DIR}/scripts/download_models.py"

if [ ! -f "$DOWNLOADER" ]; then
    echo "[ERR ] ModelZoo downloader not found: $DOWNLOADER" >&2
    exit 1
fi

# Defaults
DEFAULT_OUTPUT="${SCRIPT_DIR}/assets/models"
OUTPUT=""
SYMLINK_TARGET=""
ARGS=()

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        -h|-help)
            ARGS+=("-h")
            shift
            ;;
        --manifest=*)
            ARGS+=("$1")
            shift
            ;;
        --manifest)
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --output=*)
            OUTPUT="${1#*=}"
            ARGS+=("$1")
            shift
            ;;
        --output)
            OUTPUT="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --symlink_target_path=*)
            SYMLINK_TARGET="${1#*=}"
            shift
            ;;
        --symlink_target_path)
            SYMLINK_TARGET="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Ensure OUTPUT has a value
if [ -z "$OUTPUT" ]; then
    OUTPUT="$DEFAULT_OUTPUT"
fi

# Ensure ARGS uses the desired output directory (always download into `OUTPUT`)
CLEAN_ARGS=()
skip_next=0
for a in "${ARGS[@]}"; do
    if [ "$skip_next" -eq 1 ]; then
        skip_next=0
        continue
    fi
    case "$a" in
        --output)
            skip_next=1
            ;;
        --output=*)
            ;;
        *)
            CLEAN_ARGS+=("$a")
            ;;
    esac
done
if [ -n "$SYMLINK_TARGET" ]; then
    DOWNLOAD_TO="$SYMLINK_TARGET"
else
    DOWNLOAD_TO="$OUTPUT"
fi
ARGS=("${CLEAN_ARGS[@]}" "--output" "$DOWNLOAD_TO")

# Ensure required Python dependency is available
python3 -c "import requests" 2>/dev/null || python3 -m pip install --quiet requests

# Run downloader
python3 "$DOWNLOADER" "${ARGS[@]}"
rc=$?

if [ $rc -ne 0 ]; then
    exit $rc
fi

# Post-processing: if symlink target was provided, ensure assets output is a symlink
# Skip if --dry-run or --list was passed (no actual download occurred)
_is_dry=0
for _a in "${ARGS[@]}"; do
    [[ "$_a" == "--dry-run" || "$_a" == "--list" ]] && _is_dry=1 && break
done

if [ -n "$SYMLINK_TARGET" ] && [ "$_is_dry" -eq 0 ]; then
    if [ -L "$OUTPUT" ] || [ -d "$OUTPUT" ]; then
        rm -rf "$OUTPUT"
    fi
    mkdir -p "$(dirname "$OUTPUT")" || true
    ln -s "$(readlink -f "$SYMLINK_TARGET")" "$OUTPUT"
    echo "[INFO] Created symbolic link: $OUTPUT -> $(readlink -f "$SYMLINK_TARGET")"
fi

exit 0
