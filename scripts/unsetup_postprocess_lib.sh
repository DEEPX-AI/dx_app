#!/bin/bash
# Script to remove library path configuration for libdxapp_*_postprocess.so
# Usage: ./scripts/unsetup_postprocess_lib.sh [--session|--system] [--force]

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}/..")

# color env settings
source ${SCRIPT_DIR}/color_env.sh
source ${SCRIPT_DIR}/common_util.sh

LIB_DIR="${DX_APP_PATH}/lib"
FORCE_MODE=false

help() {
    echo -e "Usage: ${COLOR_CYAN}$0 [OPTIONS]${COLOR_RESET}"
    echo -e "Remove library path configuration for libdxapp_*_postprocess.so"
    echo -e ""
    echo -e "${COLOR_BOLD}Options:${COLOR_RESET}"
    echo -e "  ${COLOR_GREEN}--help${COLOR_RESET}       Display this help message and exit."
    echo -e "  ${COLOR_GREEN}--session${COLOR_RESET}    Remove ${LIB_DIR} from LD_LIBRARY_PATH (current session)."
    echo -e "  ${COLOR_GREEN}--system${COLOR_RESET}     Remove libs from /usr/local/lib and run ldconfig."
    echo -e "  ${COLOR_GREEN}--force${COLOR_RESET}      Skip confirmation prompt (use with --system)."
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}source $0 --session${COLOR_RESET}    # Note: use 'source' to modify current shell"
    echo -e "  ${COLOR_YELLOW}$0 --system${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --system --force${COLOR_RESET}    # Skip confirmation"
    echo -e ""
    exit 0
}

unsetup_session() {
    echo -e "${COLOR_CYAN}${COLOR_BOLD}Removing ${LIB_DIR} from LD_LIBRARY_PATH...${COLOR_RESET}"
    
    # Remove the lib directory from LD_LIBRARY_PATH
    if [[ ":$LD_LIBRARY_PATH:" == *":${LIB_DIR}:"* ]]; then
        # Remove the path (handle beginning, middle, and end cases)
        export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e "s|${LIB_DIR}:||g" -e "s|:${LIB_DIR}||g" -e "s|${LIB_DIR}||g")
        echo -e "${COLOR_GREEN}  ✓ Removed ${LIB_DIR} from LD_LIBRARY_PATH${COLOR_RESET}"
    else
        echo -e "${COLOR_YELLOW}  ⓘ ${LIB_DIR} was not found in LD_LIBRARY_PATH${COLOR_RESET}"
    fi
    
    echo ""
    echo -e "${COLOR_YELLOW}${COLOR_BOLD}Note: If you added LD_LIBRARY_PATH to ~/.bashrc or ~/.profile, remove it manually:${COLOR_RESET}"
    echo -e "${COLOR_GREEN}  # Remove this line from ~/.bashrc or ~/.profile:${COLOR_RESET}"
    echo -e "${COLOR_GREEN}  export LD_LIBRARY_PATH=${LIB_DIR}:\$LD_LIBRARY_PATH${COLOR_RESET}"
    echo ""
}

unsetup_system() {
    echo -e "${COLOR_CYAN}${COLOR_BOLD}Removing postprocess libraries from /usr/local/lib...${COLOR_RESET}"
    
    # Find and list the libraries to be removed
    libs_to_remove=$(ls /usr/local/lib/libdxapp_*_postprocess.so 2>/dev/null)
    
    if [ -z "$libs_to_remove" ]; then
        echo -e "${COLOR_YELLOW}  ⓘ No libdxapp_*_postprocess.so files found in /usr/local/lib${COLOR_RESET}"
        return 0
    fi
    
    echo -e "${TAG_INFO} The following libraries will be removed:"
    for lib in $libs_to_remove; do
        echo -e "  - $lib"
    done
    echo ""
    
    # Skip confirmation if force mode is enabled
    if [ "$FORCE_MODE" = false ]; then
        read -p "Are you sure you want to remove these libraries? [y/N]: " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo -e "${COLOR_YELLOW}  ⓘ Operation cancelled.${COLOR_RESET}"
            return 0
        fi
    fi
    
    if sudo rm -f /usr/local/lib/libdxapp_*_postprocess.so 2>/dev/null; then
        echo -e "${COLOR_GREEN}  ✓ Libraries removed from /usr/local/lib${COLOR_RESET}"
        echo -e "${COLOR_CYAN}${COLOR_BOLD}Running ldconfig...${COLOR_RESET}"
        if sudo ldconfig; then
            echo -e "${COLOR_GREEN}  ✓ ldconfig completed successfully${COLOR_RESET}"
            echo -e "${COLOR_GREEN}${COLOR_BOLD}Libraries have been removed from the system.${COLOR_RESET}"
        else
            echo -e "${COLOR_RED}${COLOR_BOLD}Failed to run ldconfig.${COLOR_RESET}"
            exit 1
        fi
    else
        echo -e "${COLOR_RED}${COLOR_BOLD}Failed to remove libraries from /usr/local/lib.${COLOR_RESET}"
        echo -e "${TAG_WARN} You may need sudo privileges."
        exit 1
    fi
}

# Parse arguments
if [ $# -eq 0 ]; then
    help
fi

# Check for --force flag first
for arg in "$@"; do
    if [ "$arg" = "--force" ] || [ "$arg" = "-f" ]; then
        FORCE_MODE=true
    fi
done

# Process main command
case "$1" in
    --help|-h)
        help
        ;;
    --session)
        unsetup_session
        ;;
    --system)
        unsetup_system
        ;;
    --force|-f)
        # If --force is the first argument, check for second argument
        if [ -n "$2" ]; then
            case "$2" in
                --system)
                    unsetup_system
                    ;;
                --session)
                    unsetup_session
                    ;;
                *)
                    echo -e "${TAG_ERROR} Invalid argument: $2"
                    help
                    ;;
            esac
        else
            echo -e "${TAG_ERROR} --force requires --session or --system"
            help
        fi
        ;;
    *)
        echo -e "${TAG_ERROR} Invalid argument: $1"
        help
        ;;
esac
