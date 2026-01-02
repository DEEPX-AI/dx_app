#!/bin/bash
# Script to set up library path for libdxapp_*_postprocess.so
# Usage: ./scripts/setup_postprocess_lib.sh [--session|--system]

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}/..")

# color env settings
source ${SCRIPT_DIR}/color_env.sh
source ${SCRIPT_DIR}/common_util.sh

LIB_DIR="${DX_APP_PATH}/lib"

help() {
    echo -e "Usage: ${COLOR_CYAN}$0 [OPTIONS]${COLOR_RESET}"
    echo -e "Set up library path for libdxapp_*_postprocess.so"
    echo -e ""
    echo -e "${COLOR_BOLD}Options:${COLOR_RESET}"
    echo -e "  ${COLOR_GREEN}--help${COLOR_RESET}       Display this help message and exit."
    echo -e "  ${COLOR_GREEN}--session${COLOR_RESET}    Export LD_LIBRARY_PATH (current session only, temporary)."
    echo -e "  ${COLOR_GREEN}--system${COLOR_RESET}     Copy libs to /usr/local/lib and run ldconfig (system-wide, permanent)."
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}source $0 --session${COLOR_RESET}    # Note: use 'source' to export in current shell"
    echo -e "  ${COLOR_YELLOW}$0 --system${COLOR_RESET}"
    echo -e ""
    exit 0
}

check_libs() {
    if [ ! -d "${LIB_DIR}" ]; then
        echo -e "${TAG_ERROR} Library directory not found: ${LIB_DIR}"
        echo -e "${TAG_INFO} Please run build.sh first."
        exit 1
    fi
    
    if ! ls ${LIB_DIR}/libdxapp_*_postprocess.so >/dev/null 2>&1; then
        echo -e "${TAG_ERROR} No libdxapp_*_postprocess.so files found in ${LIB_DIR}"
        echo -e "${TAG_INFO} Please run build.sh first."
        exit 1
    fi
}

setup_session() {
    check_libs
    echo -e "${COLOR_CYAN}${COLOR_BOLD}Setting LD_LIBRARY_PATH for current session...${COLOR_RESET}"
    export LD_LIBRARY_PATH=${LIB_DIR}:$LD_LIBRARY_PATH
    echo -e "${COLOR_GREEN}  ✓ LD_LIBRARY_PATH exported: ${LIB_DIR}${COLOR_RESET}"
    echo ""
    echo -e "${COLOR_YELLOW}${COLOR_BOLD}Note: This setting is temporary and will be lost when the terminal session ends.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}To make it permanent, add the following line to your ~/.bashrc or ~/.profile:${COLOR_RESET}"
    echo -e "${COLOR_GREEN}  export LD_LIBRARY_PATH=${LIB_DIR}:\$LD_LIBRARY_PATH${COLOR_RESET}"
    echo ""
}

setup_system() {
    check_libs
    echo -e "${COLOR_CYAN}${COLOR_BOLD}Copying libraries to /usr/local/lib...${COLOR_RESET}"
    
    if sudo cp ${LIB_DIR}/libdxapp_*_postprocess.so /usr/local/lib/ 2>/dev/null; then
        echo -e "${COLOR_GREEN}  ✓ Libraries copied to /usr/local/lib${COLOR_RESET}"
        echo -e "${COLOR_CYAN}${COLOR_BOLD}Running ldconfig...${COLOR_RESET}"
        if sudo ldconfig; then
            echo -e "${COLOR_GREEN}  ✓ ldconfig completed successfully${COLOR_RESET}"
            echo -e "${COLOR_GREEN}${COLOR_BOLD}Library path is now permanently configured (system-wide).${COLOR_RESET}"
            echo ""
            echo -e "${TAG_INFO} To uninstall, run: ${COLOR_YELLOW}./scripts/unsetup_postprocess_lib.sh --system${COLOR_RESET}"
        else
            echo -e "${COLOR_RED}${COLOR_BOLD}Failed to run ldconfig.${COLOR_RESET}"
            exit 1
        fi
    else
        echo -e "${COLOR_RED}${COLOR_BOLD}Failed to copy libraries to /usr/local/lib.${COLOR_RESET}"
        echo -e "${TAG_WARN} You may need sudo privileges."
        exit 1
    fi
}

# Parse arguments
if [ $# -eq 0 ]; then
    help
fi

case "$1" in
    --help|-h)
        help
        ;;
    --session)
        setup_session
        ;;
    --system)
        setup_system
        ;;
    *)
        echo -e "${TAG_ERROR} Invalid argument: $1"
        help
        ;;
esac
