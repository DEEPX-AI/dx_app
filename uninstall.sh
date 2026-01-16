#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR")
DOWNLOAD_DIR="$SCRIPT_DIR/download"
PROJECT_NAME=$(basename "$SCRIPT_DIR")
VENV_PATH="$PROJECT_ROOT/venv-$PROJECT_NAME"

pushd "$PROJECT_ROOT" >&2

# color env settings
source ${PROJECT_ROOT}/scripts/color_env.sh
source ${PROJECT_ROOT}/scripts/common_util.sh

ENABLE_DEBUG_LOGS=0

show_help() {
    echo -e "Usage: ${COLOR_CYAN}$(basename "$0") [OPTIONS]${COLOR_RESET}"
    echo -e ""
    echo -e "Options:"
    echo -e "  ${COLOR_GREEN}[-v|--verbose]${COLOR_RESET}                        Enable verbose (debug) logging"
    echo -e "  ${COLOR_GREEN}[-h|--help]${COLOR_RESET}                           Display this help message and exit"
    echo -e ""
    
    if [ "$1" == "error" ] && [[ ! -n "$2" ]]; then
        print_colored_v2 "ERROR" "Invalid or missing arguments."
        exit 1
    elif [ "$1" == "error" ] && [[ -n "$2" ]]; then
        print_colored_v2 "ERROR" "$2"
        exit 1
    elif [[ "$1" == "warn" ]] && [[ -n "$2" ]]; then
        print_colored_v2 "WARNING" "$2"
        return 0
    fi
    exit 0
}

uninstall_common_files() {
    print_colored_v2 "INFO" "Uninstalling common files..."
    delete_symlinks "$DOWNLOAD_DIR"
    delete_symlinks "$PROJECT_ROOT"
    delete_symlinks "${VENV_PATH}"
    delete_symlinks "${VENV_PATH}-local"
    delete_dir "${VENV_PATH}"
    delete_dir "${VENV_PATH}-local"
    delete_dir "${DOWNLOAD_DIR}" 
}

cleanup_pycache() {
    print_colored_v2 "INFO" "Cleaning up __pycache__ directories..."
    # Find and remove all __pycache__ directories under the project root
    if command -v find >/dev/null 2>&1; then
        find "$PROJECT_ROOT" -type d -name "__pycache__" -print0 2>/dev/null | while IFS= read -r -d '' dir; do
            print_colored_v2 "INFO" "Removing __pycache__ directory: $dir"
            rm -rf "$dir"
        done
    else
        print_colored_v2 "WARNING" "'find' command not available; skipping __pycache__ cleanup"
    fi
}

cleanup_pyc_files() {
    print_colored_v2 "INFO" "Cleaning up stray *.pyc files..."
    # Find and remove any loose .pyc files under the project root (outside __pycache__)
    if command -v find >/dev/null 2>&1; then
        find "$PROJECT_ROOT" -type f -name "*.pyc" -print0 2>/dev/null | while IFS= read -r -d '' file; do
            print_colored_v2 "INFO" "Removing pyc file: $file"
            rm -f "$file"
        done
    else
        print_colored_v2 "WARNING" "'find' command not available; skipping *.pyc cleanup"
    fi
}

cleanup_pytest_cache() {
    print_colored_v2 "INFO" "Cleaning up .pytest_cache directories..."
    # Find and remove all .pytest_cache directories under the project root
    if command -v find >/dev/null 2>&1; then
        find "$PROJECT_ROOT" -type d -name ".pytest_cache" -print0 2>/dev/null | while IFS= read -r -d '' dir; do
            print_colored_v2 "INFO" "Removing .pytest_cache directory: $dir"
            rm -rf "$dir"
        done
    else
        print_colored_v2 "WARNING" "'find' command not available; skipping .pytest_cache cleanup"
    fi
}

uninstall_system_libraries() {
    # Remove libdxapp_*_postprocess.so from /usr/local/lib using unsetup script
    print_colored_v2 "INFO" "Checking for system-installed postprocess libraries..."
    
    local unsetup_script="${PROJECT_ROOT}/scripts/unsetup_postprocess_lib.sh"
    
    if [ -f "$unsetup_script" ]; then
        # Use the unsetup script with --force to skip confirmation
        "$unsetup_script" --system --force
    else
        print_colored_v2 "WARNING" "unsetup_postprocess_lib.sh not found, attempting direct removal..."
        
        local libs_to_remove=$(ls /usr/local/lib/libdxapp_*_postprocess.so 2>/dev/null)
        
        if [ -z "$libs_to_remove" ]; then
            print_colored_v2 "INFO" "No libdxapp_*_postprocess.so files found in /usr/local/lib"
            return 0
        fi
        
        print_colored_v2 "INFO" "Found postprocess libraries in /usr/local/lib:"
        for lib in $libs_to_remove; do
            echo "  - $lib"
        done
        
        print_colored_v2 "INFO" "Removing libdxapp_*_postprocess.so from /usr/local/lib..."
        if sudo rm -f /usr/local/lib/libdxapp_*_postprocess.so 2>/dev/null; then
            print_colored_v2 "INFO" "Libraries removed from /usr/local/lib"
            print_colored_v2 "INFO" "Running ldconfig to update library cache..."
            if sudo ldconfig; then
                print_colored_v2 "INFO" "ldconfig completed successfully"
            else
                print_colored_v2 "WARNING" "Failed to run ldconfig"
            fi
        else
            print_colored_v2 "WARNING" "Failed to remove libraries from /usr/local/lib (may require sudo)"
        fi
    fi
}

uninstall_project_specific_files() {
    print_colored_v2 "INFO" "Uninstalling ${PROJECT_NAME} specific files..."
    delete_dir "build_*/"
    delete_dir "artifacts/"
    delete_dir "assets/"
    delete_dir "bin/"
    delete_dir "extern/pybind11/"
    delete_dir "include/"
    delete_dir "lib/"
    delete_dir "result*.jpg"
}

main() {
    echo "Uninstalling ${PROJECT_NAME} ..."

    # Remove symlinks from DOWNLOAD_DIR and PROJECT_ROOT for 'Common' Rules
    uninstall_common_files

    # Remove system-installed postprocess libraries
    uninstall_system_libraries

    # Uninstall the project specific files
    uninstall_project_specific_files

    # Cleanup Python bytecode caches
    cleanup_pycache

    # Cleanup stray .pyc files
    cleanup_pyc_files

    # Cleanup pytest cache directories
    cleanup_pytest_cache

    echo "Uninstalling ${PROJECT_NAME} done"
}

# parse args
for i in "$@"; do
    case "$1" in
        -v|--verbose)
            ENABLE_DEBUG_LOGS=1
            ;;
        -h|--help)
            show_help
            ;;
        *)
            show_help "error" "Invalid option '$1'"
            ;;
    esac
    shift
done

main

popd >&2

exit 0
