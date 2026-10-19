#!/usr/bin/env bash
# Thin wrapper: delegate all work to scripts/download_models.py
SCRIPT_DIR=$(realpath "$(dirname "$0")")
source ${SCRIPT_DIR}/scripts/color_env.sh || true
source ${SCRIPT_DIR}/scripts/common_util.sh || true

DOWNLOADER="${SCRIPT_DIR}/scripts/download_models.py"

if [ ! -f "$DOWNLOADER" ]; then
    echo "[DXAPP] [ERROR] ModelZoo downloader not found: $DOWNLOADER" >&2
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
# A trailing slash makes every test below follow a symlink: "-L link/" is
# false for a valid link and for a dangling one alike. Strip it (but keep "/").
while [ "${#OUTPUT}" -gt 1 ] && [ "${OUTPUT%/}" != "$OUTPUT" ]; do
    OUTPUT="${OUTPUT%/}"
done

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
# Where to download. An existing OUTPUT is never deleted or replaced:
#   - a dangling symlink is an error (fix or remove the link, then re-run);
#   - a symlink that resolves, or a real directory holding files, is used
#     in place (downloaded into) and --symlink_target_path is not used;
#   - only a missing OUTPUT, or an empty real directory, becomes a symlink
#     to --symlink_target_path after a successful download.
MAKE_LINK=0
if [ -L "$OUTPUT" ] && [ ! -e "$OUTPUT" ]; then
    echo "[DXAPP] [ERROR] $OUTPUT is a dangling symlink (-> $(readlink "$OUTPUT"))." >&2
    echo "[DXAPP] [ERROR] Fix the link or remove it, then re-run." >&2
    exit 1
fi
if [ -e "$OUTPUT" ] && [ ! -d "$OUTPUT" ]; then
    echo "[DXAPP] [ERROR] $OUTPUT exists and is not a directory." >&2
    exit 1
fi
if [ -d "$OUTPUT" ] && { [ ! -r "$OUTPUT" ] || [ ! -x "$OUTPUT" ]; }; then
    # `ls -A` below would print nothing for it and take it for empty.
    echo "[DXAPP] [ERROR] $OUTPUT is a directory this user cannot read; fix its permissions or pass another --output." >&2
    exit 1
fi
if [ -z "$SYMLINK_TARGET" ]; then
    DOWNLOAD_TO="$OUTPUT"
elif [ -L "$OUTPUT" ]; then
    DOWNLOAD_TO="$OUTPUT"
    echo "[DXAPP] [INFO] $OUTPUT is a symlink (-> $(readlink "$OUTPUT")); downloading through it"
elif [ -d "$OUTPUT" ] && [ -n "$(ls -A "$OUTPUT")" ]; then
    DOWNLOAD_TO="$OUTPUT"
    echo "[DXAPP] [INFO] $OUTPUT is an existing directory with files; used in place (not linked to $SYMLINK_TARGET)"
else
    DOWNLOAD_TO="$SYMLINK_TARGET"
    MAKE_LINK=1
fi
ARGS=("${CLEAN_ARGS[@]}" "--output" "$DOWNLOAD_TO")

# The Python that runs the downloader (U-76); the first match wins:
#   $DXAPP_SETUP_PYTHON  an explicit choice (a path or a command name)
#   $VIRTUAL_ENV         the active virtualenv
#   ../venv-dx-runtime   the dx-runtime venv (dx-runtime/install.sh)
#   python3              the first on PATH
# The venvs come before PATH because a conda python3 on PATH (3.13+) verifies
# TLS in X.509 strict mode, which a TLS-inspecting proxy's certificate chain
# can fail ("Missing Authority Key Identifier"). download_models.py names
# the ways past that when it happens.
pick_downloader_python() {
    if [ -n "${DXAPP_SETUP_PYTHON:-}" ]; then
        # A file that is executable, or an executable found on PATH: not a
        # directory, and not a shell builtin or function (`command -v` finds those).
        if { [ -f "${DXAPP_SETUP_PYTHON}" ] && [ -x "${DXAPP_SETUP_PYTHON}" ]; } \
            || type -P "${DXAPP_SETUP_PYTHON}" >/dev/null; then
            echo "${DXAPP_SETUP_PYTHON}"
            return 0
        fi
        echo "[DXAPP] [ERROR] DXAPP_SETUP_PYTHON=${DXAPP_SETUP_PYTHON} is not an executable Python." >&2
        return 1
    fi
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        echo "${VIRTUAL_ENV}/bin/python"
        return 0
    fi
    # -s: normalise the path, but do not resolve the venv's python symlink
    # (resolving it would leave the venv).
    if [ -x "${SCRIPT_DIR}/../venv-dx-runtime/bin/python" ]; then
        realpath -s "${SCRIPT_DIR}/../venv-dx-runtime/bin/python"
        return 0
    fi
    echo python3
}

DOWNLOADER_PYTHON=$(pick_downloader_python) || exit 1
echo "[DXAPP] [INFO] Downloader Python: ${DOWNLOADER_PYTHON} ($("${DOWNLOADER_PYTHON}" -c \
    'import ssl, sys; print("Python %s, %s" % (sys.version.split()[0], ssl.OPENSSL_VERSION))' \
    2>/dev/null || echo "version unknown"))"

# Ensure required Python dependency is available (in that interpreter only)
"${DOWNLOADER_PYTHON}" -c "import requests" 2>/dev/null || "${DOWNLOADER_PYTHON}" -m pip install --quiet requests

# Run downloader
"${DOWNLOADER_PYTHON}" "$DOWNLOADER" "${ARGS[@]}"
rc=$?

if [ $rc -ne 0 ]; then
    exit $rc
fi

# Post-processing: link a missing (or empty) OUTPUT to the symlink target
# Skip if --dry-run or --list was passed (no actual download occurred)
_is_dry=0
for _a in "${ARGS[@]}"; do
    [[ "$_a" == "--dry-run" || "$_a" == "--list" ]] && _is_dry=1 && break
done

if [ "$MAKE_LINK" -eq 1 ] && [ "$_is_dry" -eq 0 ]; then
    # OUTPUT is absent or an empty real directory (checked above); rmdir
    # refuses anything else.
    if [ -d "$OUTPUT" ] && [ ! -L "$OUTPUT" ]; then
        rmdir "$OUTPUT" || { echo "[DXAPP] [ERROR] $OUTPUT is no longer empty; not replacing it with a symlink." >&2; exit 1; }
    fi
    mkdir -p "$(dirname "$OUTPUT")" || true
    ln -sn "$(readlink -f "$SYMLINK_TARGET")" "$OUTPUT" || {
        echo "[DXAPP] [ERROR] Could not create the symbolic link $OUTPUT -> $(readlink -f "$SYMLINK_TARGET")." >&2
        echo "[DXAPP] [ERROR] The models were downloaded to $SYMLINK_TARGET; link or move them to $OUTPUT." >&2
        exit 1
    }
    echo "[DXAPP] [INFO] Created symbolic link: $OUTPUT -> $(readlink -f "$SYMLINK_TARGET")"
fi

exit 0
