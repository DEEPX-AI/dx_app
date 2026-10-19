#!/bin/bash
# =============================================================================
# gui_env.sh - quiet OpenCV HighGUI start-up warnings for dx_app examples
# =============================================================================
# Source this file, then call:
#
#   dxapp_prepare_gui_env            # export env for the current shell
#   dxapp_link_cv2_qt_fonts <python> # fix the font dir of that python's cv2
#
# Neither function overrides a value the user already set, and both are safe
# to call more than once. The examples apply the same fixes in code before
# their first window, so a binary or script started by hand is covered too.
# =============================================================================

# System font directories, most preferred first.
DXAPP_GUI_FONT_DIRS=(
    "/usr/share/fonts/truetype/dejavu"
    "/usr/share/fonts/truetype/liberation"
    "/usr/share/fonts/TTF"
    "/usr/share/fonts"
)

# Print the first system font directory that exists; return 1 when none does.
dxapp_system_font_dir() {
    local font_dir
    for font_dir in "${DXAPP_GUI_FONT_DIRS[@]}"; do
        if [ -d "${font_dir}" ]; then
            echo "${font_dir}"
            return 0
        fi
    done
    return 1
}

# GTK prints "dbind-WARNING: Couldn't connect to accessibility bus" when the
# AT-SPI socket is missing. Skip the bridge only in that case, so a running
# screen reader keeps working.
dxapp_prepare_gui_env() {
    # Socket the AT-SPI bridge would connect to for this user.
    local atspi_bus="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/at-spi/bus"
    if [ -z "${NO_AT_BRIDGE+x}" ] && [ ! -S "${atspi_bus}" ]; then
        export NO_AT_BRIDGE=1
    fi
}

# opencv-python sets QT_QPA_FONTDIR to <cv2>/qt/fonts, which the wheel does not
# ship, so Qt prints "QFontDatabase: Cannot find font directory" per font
# lookup. Link that path to a system font directory. Returns 0 when nothing
# needs doing (no python, no cv2, headless wheel, or the dir already exists).
dxapp_link_cv2_qt_fonts() {
    local python_exec="${1:-python3}"
    # Directory cv2 points QT_QPA_FONTDIR at; empty when cv2 has no Qt.
    local cv2_qt_fonts
    # System font directory the link targets.
    local sys_font_dir

    command -v "${python_exec}" >/dev/null 2>&1 || [ -x "${python_exec}" ] || return 0
    cv2_qt_fonts=$("${python_exec}" - 2>/dev/null <<'PY'
import os
try:
    import cv2
except Exception:
    raise SystemExit(0)
qt_dir = os.path.join(os.path.dirname(cv2.__file__), "qt")
if os.path.isdir(os.path.join(qt_dir, "plugins")):
    print(os.path.join(qt_dir, "fonts"))
PY
)
    [ -n "${cv2_qt_fonts}" ] || return 0
    [ -e "${cv2_qt_fonts}" ] && return 0

    if ! sys_font_dir=$(dxapp_system_font_dir); then
        print_colored "No system fonts for OpenCV Qt. Install them with: sudo apt install fonts-dejavu-core" "WARNING"
        return 0
    fi
    if ln -s "${sys_font_dir}" "${cv2_qt_fonts}" 2>/dev/null; then
        print_colored "[OK] OpenCV Qt fonts: ${cv2_qt_fonts} -> ${sys_font_dir}" "INFO"
    else
        print_colored "Cannot link ${cv2_qt_fonts} (read-only site-packages?). Examples fall back to ${sys_font_dir} at run time." "WARNING"
    fi
    return 0
}
