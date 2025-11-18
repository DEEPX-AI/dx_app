#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

# color env settings
source "${SCRIPT_DIR}/color_env.sh"
source "${SCRIPT_DIR}/common_util.sh"

# Global variables for script configuration
DEFAULT_MIN_PY_VERSION="3.8.10"

# ---
## Usage Information
# ---
usage() {
    echo -e "Usage: ${COLOR_CYAN}$0 [OPTIONS]${COLOR_RESET}"
    echo -e "Install a specified Python version and optionally set up a virtual environment."
    echo -e ""
    echo -e "${COLOR_BOLD}Options:${COLOR_RESET}"
    echo -e "  ${COLOR_GREEN}--python_version=<VERSION>${COLOR_RESET}  Specify the Python version to install (e.g., 3.10.4)."
    echo -e "                                Default Minimum supported version: ${DEFAULT_MIN_PY_VERSION}."
    echo -e "                                If not specified:"
    echo -e "                                  - For Ubuntu 20.04+, the OS default Python 3 will be used."
    echo -e "                                  - For Ubuntu 18.04, Python ${DEFAULT_MIN_PY_VERSION} "
    echo -e "                                    (or the value specified by the '--min_py_version' option) will be source-built."
    echo -e ""
    echo -e "  ${COLOR_GREEN}--min_py_version=<VERSION>${COLOR_RESET}  Specify the minimum Python version. (default: ${DEFAULT_MIN_PY_VERSION})"
    echo -e ""
    echo -e "  ${COLOR_GREEN}--venv_path=<PATH>${COLOR_RESET}          Specify the path for the virtual environment."
    echo -e "                                  - If this option is omitted, no virtual environment will be created."
    echo -e ""
    echo -e "  ${COLOR_GREEN}--symlink_target_path=<PATH>${COLOR_RESET} Specify the actual path where the virtual environment will be created."
    echo -e "                                  - If specified, a symbolic link will be created at --venv_path pointing to this path."
    echo -e "                                  - Only works when --venv_path is also specified."
    echo -e ""
    echo -e "  ${COLOR_GREEN}--system-site-packages${COLOR_RESET}      Set venv '--system-site-packages' option."    
    echo -e "                                  - This option is applied only when venv is created. If you use '-venv-reuse', it is ignored. "
    echo -e ""
    echo -e "  ${COLOR_GREEN}-f | --venv-force-remove${COLOR_RESET}    If specified, force remove existing virtual environment at --venv_path before creation."
    echo -e "  ${COLOR_GREEN}-r | --venv-reuse${COLOR_RESET}           If specified, reuse existing virtual environment at --venv_path if it's valid, skipping creation."
    echo -e ""
    echo -e "  ${COLOR_GREEN}--help${COLOR_RESET}                      Display this help message and exit."
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 # Installs default Python, but no venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --python_version=3.10.4 --venv_path=./my_venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --python_version=3.9.18  # Installs Python, but no venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=./my_venv # Installs default Python, creates venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=./existing_venv --venv-reuse # Reuse existing venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=./old_venv --venv-force-remove # Force remove and recreate venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=./my_venv --symlink_target_path=/tmp/actual_venv # Create venv at /tmp/actual_venv with symlink at ./my_venv${COLOR_RESET}"
    echo -e ""
    exit 0
}

# ---
## Check if Python version is already installed and usable, or if a higher suitable version exists.
# Returns 0 if installed and usable, 1 otherwise.
# Outputs the executable path to stdout if found and usable.
# Logs to stderr.
# Arguments:
#   $1: OS_ID - The operating system ID (ubuntu, debian, etc.)
#   $2: OS_VERSION - Current OS release version (e.g., "20.04" for Ubuntu, "12" for Debian)
#   $3: REQUESTED_PY_VERSION (optional) - The specific Python version requested (e.g., "3.8.2").
#                                         If empty, means OS default/MIN_PY_VERSION is implied.
#   $4: MIN_REQUIRED_PY_VERSION - The absolute minimum Python version required by the script.
# ---
is_python_installed() {
    local OS_ID="${1}"
    local OS_VERSION="${2}"
    local REQUESTED_PY_VERSION="${3}"
    local MIN_REQUIRED_PY_VERSION="${4}"
    local PYTHON_EXECUTABLES=("python3.12" "python3.11" "python3.10" "python3.9" "python3.8" "python3") # Order matters: higher to lower

    local REQ_VER_NUM=0
    if [ -n "${REQUESTED_PY_VERSION}" ]; then
        REQ_VER_NUM=$(printf "%02d%02d%02d" $(echo "${REQUESTED_PY_VERSION}" | tr '.' ' '))
    fi
    local MIN_REQ_VER_NUM=$(printf "%02d%02d%02d" $(echo "${MIN_REQUIRED_PY_VERSION}" | tr '.' ' '))

    echo -e "${TAG_INFO} Checking for existing Python installations that meet requirements..." >&2

    for cmd in "${PYTHON_EXECUTABLES[@]}"; do
        local check_path="/usr/bin/${cmd}" # Default apt path
        local source_path="/usr/local/bin/${cmd}" # Default source path

        local current_exec=""
        local current_version_full=""
        local current_version_num=0

        # Try apt path first for modern OS versions, or if command exists
        if { [ "$OS_ID" = "ubuntu" ] && { [ "$OS_VERSION" = "24.04" ] || [ "$OS_VERSION" = "22.04" ] || [ "$OS_VERSION" = "20.04" ]; }; } || \
           { [ "$OS_ID" = "debian" ] && { [ "$OS_VERSION" = "12" ] || [ "$OS_VERSION" = "11" ]; }; } || \
           command -v "${cmd}" &>/dev/null; then
            if [ -x "${check_path}" ]; then
                current_exec="${check_path}"
                current_version_full=$("${current_exec}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "unknown")
            elif command -v "${cmd}" &>/dev/null; then # Fallback to PATH check if not at /usr/bin
                current_exec=$(command -v "${cmd}")
                current_version_full=$("${current_exec}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "unknown")
            fi
        fi

        # If not found via apt-like paths or command, try source-built path explicitly
        if [ -z "${current_exec}" ] && [ -x "${source_path}" ]; then
            current_exec="${source_path}"
            current_version_full=$("${current_exec}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")' 2>/dev/null || echo "unknown")
        fi

        if [ -n "${current_exec}" ] && [ "${current_version_full}" != "unknown" ]; then
            current_version_num=$(printf "%02d%02d%02d" $(echo "${current_version_full}" | tr '.' ' '))
            echo -e "${TAG_INFO} Detected Python ${current_version_full} at ${current_exec}." >&2

            # Check if detected version meets the minimum script requirement
            if [ "${current_version_num}" -ge "${MIN_REQ_VER_NUM}" ]; then
                # If a specific version was requested (--python_version), check if the detected one is suitable.
                # Suitable means:
                # 1. Detected version's major.minor matches requested major.minor AND detected version >= requested version (e.g., 3.8.10 for 3.8.2)
                # 2. Or, if requested version is empty (meaning "any suitable"), then this detected higher version is fine.
                # (disabled condition) 3. Or, if requested version is explicitly provided, and detected is a higher major.minor version.
                
                local current_major_minor=$(echo "${current_version_full}" | cut -d. -f1,2)
                local requested_major_minor=$(echo "${REQUESTED_PY_VERSION}" | cut -d. -f1,2)

                if [ -z "${REQUESTED_PY_VERSION}" ]; then # No specific version requested, any valid is fine
                    echo -e "${TAG_INFO} Python ${current_version_full} is suitable as no specific version was requested. Using this version." >&2
                    echo "${current_exec}" # Output the usable executable path
                    return 0
                elif [ "${current_major_minor}" = "${requested_major_minor}" ] && [ "${current_version_num}" -ge "${REQ_VER_NUM}" ]; then
                    echo -e "${TAG_INFO} Python ${current_version_full} matches requested major.minor and is compatible (>= requested ${REQUESTED_PY_VERSION}). Using this version." >&2
                    echo "${current_exec}" # Output the usable executable path
                    return 0
                # (disabled condition)
                # elif [ "${current_version_num}" -ge "${REQ_VER_NUM}" ]; then # Detected version is higher than requested, even if major.minor differs
                #     echo -e "${TAG_INFO} Python ${current_version_full} is a higher version than requested (${REQUESTED_PY_VERSION}) and meets minimums. Using this version." >&2
                #     echo "${current_exec}" # Output the usable executable path
                #     return 0
                fi
            else
                echo -e "${TAG_WARN} Python ${current_version_full} found, but it is below the minimum required version (${MIN_REQUIRED_PY_VERSION})." >&2
            fi
        fi
    done

    echo -e "${TAG_INFO} No suitable Python installation found on the system." >&2
    return 1 # No suitable Python found
}

# ---
## Install Python and its dependencies (dev, venv)
# ---
# Arguments:
#   $1: TARGET_INSTALL_PY_VERSION (optional) - The specific Python version to install.
#         If empty, the OS default Python 3 version will be installed for Ubuntu 20.04+ and Debian 11+.
#         For Ubuntu 18.04, MIN_PY_VERSION will be used if TARGET_INSTALL_PY_VERSION is empty.
#   $2: OS_ID - The operating system ID (ubuntu, debian, etc.)
#   $3: OS_VERSION - The current OS release version (e.g., "20.04" for Ubuntu, "12" for Debian)
install_python_and_dependencies() {
    local TARGET_INSTALL_PY_VERSION="${1}"
    local OS_ID="${2}"
    local OS_VERSION="${3}"

    # Temporarily disable 'set -e' for controlled error handling and 'set -x' for cleaner output capture
    local OPT_E_STATE="$-"
    local OPT_X_STATE="$-"
    case "${OPT_E_STATE}" in *e*) set +e;; esac
    case "${OPT_X_STATE}" in *x*) set +x;; esac

    exec 3>&1 # Save stdout to fd 3
    exec >&2  # Redirect stdout to stderr

    echo -e "${TAG_INFO} Starting Python installation/dependency checks for ${TARGET_INSTALL_PY_VERSION:-default} on ${OS_ID} ${OS_VERSION}..."

    local DX_PYTHON_EXEC_OUT="" # This will hold the command to execute the installed python
    local INSTALL_STATUS=0
    local PY_MAJOR_MINOR=""

    if [ -n "${TARGET_INSTALL_PY_VERSION}" ]; then
        PY_MAJOR_MINOR=$(echo "${TARGET_INSTALL_PY_VERSION}" | cut -d. -f1,2)
    fi

    # Determine the Python executable name based on requested version or default
    local PYTHON_EXE_NAME=""
    if [ -n "${PY_MAJOR_MINOR}" ]; then
        PYTHON_EXE_NAME="python${PY_MAJOR_MINOR}"
    else
        PYTHON_EXE_NAME="python3" # Default for modern OS versions
    fi

    # Check if Python is already installed OR if a higher suitable version exists
    local IS_INSTALLED_RESULT
    IS_INSTALLED_RESULT=$(is_python_installed "${OS_ID}" "${OS_VERSION}" "${TARGET_INSTALL_PY_VERSION}" "${MIN_PY_VERSION}")
    
    if [ -n "${IS_INSTALLED_RESULT}" ]; then
        DX_PYTHON_EXEC_OUT="${IS_INSTALLED_RESULT}"
        echo -e "${TAG_SKIP} A suitable Python installation is already present (${DX_PYTHON_EXEC_OUT})."
        echo -e "${TAG_INFO} Ensuring required development and venv packages for ${DX_PYTHON_EXEC_OUT} are installed."
        
        # Install dev/venv packages even if main interpreter is skipped
        # For Ubuntu and Debian modern versions (Ubuntu 20.04+, Debian 11+)
        if { [ "$OS_ID" = "ubuntu" ] && { [ "$OS_VERSION" = "24.04" ] || [ "$OS_VERSION" = "22.04" ] || [ "$OS_VERSION" = "20.04" ]; }; } || \
           { [ "$OS_ID" = "debian" ] && { [ "$OS_VERSION" = "12" ] || [ "$OS_VERSION" = "11" ]; }; }; then
            local TARGET_MAJOR_MINOR=$(echo "${DX_PYTHON_EXEC_OUT}" | sed -n 's/.*python\([0-9]\.[0-9]\+\).*/\1/p')
            if [ -z "${TARGET_MAJOR_MINOR}" ]; then TARGET_MAJOR_MINOR="3"; fi # Fallback for 'python3'

            # Check if dev and venv packages are already installed
            if dpkg -s "python${TARGET_MAJOR_MINOR}-dev" >/dev/null 2>&1 && dpkg -s "python${TARGET_MAJOR_MINOR}-venv" >/dev/null 2>&1; then
                echo -e "${TAG_SKIP} Python${TARGET_MAJOR_MINOR}-dev and python${TARGET_MAJOR_MINOR}-venv are already installed. Skipping."
            else
                if ! sudo apt-get update; then
                    print_colored "Failed to update apt repositories" "ERROR"
                    INSTALL_STATUS=1
                elif ! sudo apt install -y gnupg gpg-agent; then
                    print_colored "Failed to install gnupg gpg-agent." "ERROR"
                    INSTALL_STATUS=1
                # Only add deadsnakes PPA for Ubuntu (not needed for Debian)
                elif [ "$OS_ID" = "ubuntu" ] && ! sudo add-apt-repository -y ppa:deadsnakes/ppa; then
                    print_colored "Failed to add deadsnakes PPA." "ERROR"
                    INSTALL_STATUS=1
                elif ! sudo apt-get update; then
                    print_colored "Failed to update apt repositories after adding PPA." "ERROR"
                    INSTALL_STATUS=1
                elif ! sudo apt-get install -y python${TARGET_MAJOR_MINOR}-dev python${TARGET_MAJOR_MINOR}-venv; then
                    print_colored "Failed to install python${TARGET_MAJOR_MINOR}-dev and/or python${TARGET_MAJOR_MINOR}-venv." "ERROR"
                    INSTALL_STATUS=1
                fi
            fi
        fi
        # Source-built Python usually includes venv, dev headers etc. if built correctly.
        # No extra apt installs needed here for Ubuntu 18.04 source builds.
    else
        echo -e "${TAG_INFO} No suitable Python installation found. Proceeding with requested installation."

        # Modern Ubuntu and Debian versions (Ubuntu 20.04+, Debian 11+)
        if { [ "$OS_ID" = "ubuntu" ] && { [ "$OS_VERSION" = "24.04" ] || [ "$OS_VERSION" = "22.04" ] || [ "$OS_VERSION" = "20.04" ]; }; } || \
           { [ "$OS_ID" = "debian" ] && { [ "$OS_VERSION" = "12" ] || [ "$OS_VERSION" = "11" ]; }; }; then
            if [ -n "${TARGET_INSTALL_PY_VERSION}" ]; then
                echo -e "${TAG_INFO} Installing python ${PY_MAJOR_MINOR} version using apt..."
                if ! sudo apt-get update; then
                    print_colored "Failed to update apt repositories" "ERROR"
                    INSTALL_STATUS=1
                elif ! sudo apt install -y gnupg gpg-agent; then
                    print_colored "Failed to install gnupg gpg-agent." "ERROR"
                    INSTALL_STATUS=1
                # Only add deadsnakes PPA for Ubuntu (not needed for Debian)
                elif [ "$OS_ID" = "ubuntu" ] && ! sudo add-apt-repository -y ppa:deadsnakes/ppa; then
                    print_colored "Failed to add deadsnakes PPA." "ERROR"
                    INSTALL_STATUS=1
                elif ! sudo apt-get update; then
                    print_colored "Failed to update apt repositories after adding PPA." "ERROR"
                    INSTALL_STATUS=1
                elif ! sudo apt-get install -y python${PY_MAJOR_MINOR} python${PY_MAJOR_MINOR}-dev python${PY_MAJOR_MINOR}-venv; then
                    print_colored "apt installation failed for python${PY_MAJOR_MINOR}." "ERROR"
                    INSTALL_STATUS=1
                fi
                DX_PYTHON_EXEC_OUT="python${PY_MAJOR_MINOR}"
            else
                echo -e "${TAG_INFO} Installing OS default python3 version and dependencies using apt..."
                if ! sudo apt-get update; then
                    print_colored "Failed to update apt repositories" "ERROR"
                    INSTALL_STATUS=1
                elif ! sudo apt-get install -y python3 python3-dev python3-venv; then
                    print_colored "apt installation failed for python3." "ERROR"
                    INSTALL_STATUS=1
                fi
                DX_PYTHON_EXEC_OUT="python3"
            fi
        # Ubuntu 18.04 - requires source build for modern Python versions
        elif [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "18.04" ]; then
            if [ ! -n "${TARGET_INSTALL_PY_VERSION}" ]; then
                TARGET_INSTALL_PY_VERSION=${MIN_PY_VERSION}
            fi
            PY_MAJOR_MINOR=$(echo "${TARGET_INSTALL_PY_VERSION}" | cut -d. -f1,2) # Ensure PY_MAJOR_MINOR is set for source build
            echo -e "${TAG_INFO} Installing python ${TARGET_INSTALL_PY_VERSION} version using source build..."
            if ! sudo apt-get update; then INSTALL_STATUS=1; fi
            if [ ${INSTALL_STATUS} -eq 0 ] && \
                ! sudo apt-get install -y --no-install-recommends \
                build-essential \
                wget \
                curl \
                ca-certificates \
                libssl-dev \
                zlib1g-dev \
                libncurses5-dev \
                libncursesw5-dev \
                libreadline-dev \
                libsqlite3-dev \
                libgdbm-dev \
                libdb5.3-dev \
                libbz2-dev \
                libexpat1-dev \
                liblzma-dev \
                tk-dev \
                libffi-dev \
                uuid-dev; then INSTALL_STATUS=1; fi

            if [ ${INSTALL_STATUS} -eq 0 ]; then
                # Ensure we are in a clean directory before attempting to build
                local BUILD_DIR="${SCRIPT_DIR}/python_build"
                mkdir -p "${BUILD_DIR}"
                pushd "${BUILD_DIR}" >/dev/null # Push current directory, suppress output

                if ! wget --no-check-certificate "https://www.python.org/ftp/python/${TARGET_INSTALL_PY_VERSION}/Python-${TARGET_INSTALL_PY_VERSION}.tgz" || \
                    ! tar xvf "Python-${TARGET_INSTALL_PY_VERSION}.tgz" || \
                    ! (cd "Python-${TARGET_INSTALL_PY_VERSION}" && ./configure --enable-optimizations && make -j$(nproc) && sudo make altinstall); then
                    print_colored "Source build for Python ${TARGET_INSTALL_PY_VERSION} failed." "ERROR"
                    INSTALL_STATUS=1
                fi
                popd >/dev/null # Pop back to original directory
                rm -rf "${BUILD_DIR}" # Clean up build directory

                if [ ${INSTALL_STATUS} -eq 0 ]; then
                    DX_PYTHON_EXEC_OUT="python${PY_MAJOR_MINOR}" # Set executable for source build
                fi
            fi
        else
            print_colored "Unsupported OS version: $OS_ID $OS_VERSION" "ERROR"
            INSTALL_STATUS=1
        fi
    fi

    if [ ${INSTALL_STATUS} -eq 0 ]; then
        echo -e "${TAG_INFO} Python installation/dependency checks done. Resolved Python executable: ${DX_PYTHON_EXEC_OUT}"
        echo "${DX_PYTHON_EXEC_OUT}" >&3 # Output the python executable path to original stdout (fd 3)
    else
        echo "" >&3 # Output empty string on failure
    fi

    # Restore original stdout and stderr
    exec 1>&3- # Restore stdout from fd 3 and close fd 3

    # Restore 'set -e' and 'set -x' if they were enabled originally
    case "${OPT_E_STATE}" in *e*) set -e;; esac
    case "${OPT_X_STATE}" in *x*) set -x;; esac

    return ${INSTALL_STATUS}
}

# ---
## Check Virtual Environment Validity
# Returns 0 if valid, 1 otherwise.
# Arguments:
#   $1: VENV_PATH_TO_CHECK - The path to the virtual environment.
# ---
function check_venv_validity() {
    local VENV_PATH_TO_CHECK="${1}"
    echo -e "${TAG_INFO} Checking virtual environment validity at ${VENV_PATH_TO_CHECK}..." >&2

    if [ ! -d "${VENV_PATH_TO_CHECK}" ]; then
        echo -e "${TAG_WARN} Venv path ${VENV_PATH_TO_CHECK} does not exist." >&2
        return 1
    fi

    if [ ! -f "${VENV_PATH_TO_CHECK}/bin/activate" ]; then
        echo -e "${TAG_WARN} Venv activate script not found: ${VENV_PATH_TO_CHECK}/bin/activate." >&2
        return 1
    fi

    if [ ! -x "${VENV_PATH_TO_CHECK}/bin/python" ]; then
        echo -e "${TAG_WARN} Venv python executable not found or not executable: ${VENV_PATH_TO_CHECK}/bin/python." >&2
        return 1
    fi

    # Test if the python executable works
    if ! "${VENV_PATH_TO_CHECK}/bin/python" -c "import sys; print('Python in venv is working!')" >/dev/null 2>&1; then
        echo -e "${TAG_WARN} Python executable in venv (${VENV_PATH_TO_CHECK}/bin/python) is not functional." >&2
        return 1
    fi

    echo -e "${TAG_SUCC} Virtual environment at ${VENV_PATH_TO_CHECK} appears to be valid." >&2
    return 0
}


# ---
## Setup Virtual Environment
# ---
# Arguments:
#   $1: DX_PYTHON_EXEC - The command to execute the installed Python (e.g., python3.10).
#   $2: VENV_PATH - The desired path for the virtual environment.
#   $3: SKIP_VENV_CREATION_FLAG - 'y' to skip venv creation, 'n' otherwise.
#   $4: VENV_MAKE_ARGS
#   $5: VENV_SYMLINK_TARGET_PATH - Optional path where the actual venv will be created (if set, VENV_PATH becomes a symlink)
setup_venv() {
    local DX_PYTHON_EXEC="${1}"
    local VENV_PATH="${2}"
    local SKIP_VENV_CREATION_FLAG="${3}"
    local VENV_MAKE_ARGS="${4}"
    local VENV_SYMLINK_TARGET_PATH="${5}"

    # Temporarily disable 'set -x' for cleaner output during venv setup steps
    local OPT_X_STATE="$-"
    case "${OPT_X_STATE}" in *x*) set +x;; esac

    if [ -z "${DX_PYTHON_EXEC}" ]; then
        print_colored "Python executable not provided to setup_venv." "ERROR" >&2
        return 1
    fi
    if [ -z "${VENV_PATH}" ]; then
        print_colored "Virtual environment path not provided to setup_venv." "ERROR" >&2
        return 1
    fi

    # Determine actual venv creation path
    local VENV_ORIGIN_DIR="${VENV_PATH}"
    
    # Convert relative path to absolute path
    if [[ "${VENV_ORIGIN_DIR}" != /* ]]; then
        VENV_ORIGIN_DIR="$(pwd)/${VENV_ORIGIN_DIR}"
    fi
    
    if [ -n "${VENV_SYMLINK_TARGET_PATH}" ]; then
        VENV_ORIGIN_DIR="${VENV_SYMLINK_TARGET_PATH}"
        echo -e "${TAG_INFO} Creating python venv to symlink target path: ${VENV_ORIGIN_DIR}"
    else
        echo -e "${TAG_INFO} Creating python venv to this path: ${VENV_ORIGIN_DIR}"
    fi

    if [ "${SKIP_VENV_CREATION_FLAG}" != "y" ]; then
        echo -e "${TAG_INFO} Setting up Virtual Environment at ${VENV_ORIGIN_DIR} using ${DX_PYTHON_EXEC}..."
        if ! "${DX_PYTHON_EXEC}" -m venv "${VENV_ORIGIN_DIR}" ${VENV_MAKE_ARGS}; then
            print_colored "Failed to create virtual environment at ${VENV_ORIGIN_DIR}." >&2
            case "${OPT_X_STATE}" in *x*) set -x;; esac # Restore set -x before returning
            return 1
        fi
    else
        echo -e "${TAG_INFO} Skipping virtual environment creation as --venv-reuse was specified and venv is valid."
    fi

    # Create symbolic link if needed
    if [ -n "${VENV_SYMLINK_TARGET_PATH}" ]; then
        echo -e "${TAG_INFO} Creating symbolic link from ${VENV_PATH} to ${VENV_ORIGIN_DIR}..."
        
        # Remove any existing symlink or directory at VENV_PATH
        if [ -e "${VENV_PATH}" ]; then
            rm -rf "${VENV_PATH}"
        fi
        
        # Ensure the parent directory exists
        mkdir -p "$(dirname "${VENV_PATH}")"
        
        # Create the symbolic link using absolute path
        local VENV_SYMLINK_TARGET_REAL_PATH
        VENV_SYMLINK_TARGET_REAL_PATH=$(readlink -f "${VENV_ORIGIN_DIR}")
        if ! ln -s "${VENV_SYMLINK_TARGET_REAL_PATH}" "${VENV_PATH}"; then
            print_colored "Failed to create symbolic link: ${VENV_PATH} -> ${VENV_SYMLINK_TARGET_REAL_PATH}" "ERROR" >&2
            case "${OPT_X_STATE}" in *x*) set -x;; esac # Restore set -x before returning
            return 1
        fi
        echo -e "${TAG_INFO} Created symbolic link: ${VENV_PATH} -> ${VENV_SYMLINK_TARGET_REAL_PATH}"
    fi

    # Activate the venv temporarily for pip operations
    echo -e "${TAG_INFO} Activating virtual environment for package upgrades..."
    if ! source "${VENV_PATH}/bin/activate"; then # Use 'source' or '.' here
        print_colored "Failed to activate virtual environment." >&2
        case "${OPT_X_STATE}" in *x*) set -x;; esac # Restore set -x before returning
        return 1
    fi

    echo -e "${TAG_INFO} Upgrading pip, wheel, and setuptools..."
    
    # Get OS information
    local OS_ID=""
    local OS_VERSION=""
    
    if [ -f /etc/os-release ]; then
        OS_ID=$(grep "^ID=" /etc/os-release | sed 's/^ID=//' | tr -d '"')
    fi
    OS_VERSION=$(lsb_release -rs)
    
    echo -e "${TAG_INFO} *** OS: ${OS_ID} ${OS_VERSION} ***"

    local PIP_INSTALL_STATUS=0
    
    # For Ubuntu 24.04 - only upgrade setuptools
    if [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "24.04" ]; then
      if ! pip install --upgrade setuptools; then PIP_INSTALL_STATUS=1; fi
    # For Debian 12 (similar to Ubuntu 24.04) - only upgrade setuptools  
    elif [ "$OS_ID" = "debian" ] && [ "$OS_VERSION" = "12" ]; then
      if ! pip install --upgrade setuptools; then PIP_INSTALL_STATUS=1; fi
    # For Ubuntu 22.04, 20.04, 18.04 and Debian 11 - upgrade pip, wheel, setuptools
    elif { [ "$OS_ID" = "ubuntu" ] && { [ "$OS_VERSION" = "22.04" ] || [ "$OS_VERSION" = "20.04" ] || [ "$OS_VERSION" = "18.04" ]; }; } || \
         { [ "$OS_ID" = "debian" ] && [ "$OS_VERSION" = "11" ]; }; then
      if ! pip install --upgrade pip wheel setuptools; then PIP_INSTALL_STATUS=1; fi
    else
      echo -e "${TAG_WARN} Unsupported OS version for specific pip upgrade rules: ${OS_ID} ${OS_VERSION}" >&2
      if ! pip install --upgrade pip wheel setuptools; then PIP_INSTALL_STATUS=1; fi # Fallback to general upgrade
    fi

    if [ ${PIP_INSTALL_STATUS} -ne 0 ]; then
        echo -e "${TAG_WARN} Pip/wheel/setuptools upgrade failed. Proceeding..." >&2
    fi

    echo -e "${TAG_INFO} Virtual environment setup complete."
    # Deactivate the venv so the script doesn't leave the current shell in the venv
    deactivate || true

    # Restore 'set -x' if it was enabled originally
    case "${OPT_X_STATE}" in *x*) set -x;; esac
    return 0
}

# ---
## Main Function
# ---
main() {
    local PYTHON_VERSION=""
    local MIN_PY_VERSION=$DEFAULT_MIN_PY_VERSION
    local VENV_PATH="" # Initialize as empty string
    local VENV_SYMLINK_TARGET_PATH=""
    local FORCE_REMOVE_VENV="n"
    local REUSE_VENV="n"
    local VENV_SYSTEM_SITE_PACKAGES_ARGS=""
    local SKIP_VENV_CREATION="n" # Flag to control venv creation in setup_venv

    # Get OS information using lsb_release (supports both Ubuntu and Debian)
    local OS_ID=""
    local OS_VERSION=""
    
    # Extract OS ID from /etc/os-release
    if [ -f /etc/os-release ]; then
        OS_ID=$(grep "^ID=" /etc/os-release | sed 's/^ID=//' | tr -d '"')
    fi
    
    # Get OS version using lsb_release
    OS_VERSION=$(lsb_release -rs)
    
    echo -e "${TAG_INFO} Detected OS: ${OS_ID} ${OS_VERSION}"

    # Parse command-line arguments
    for i in "$@"; do
        case $i in
            --python_version=*)
                PYTHON_VERSION="${i#*=}"
                ;;
            --min_py_version=*)
                MIN_PY_VERSION="${i#*=}"
                ;;
            --venv_path=*)
                VENV_PATH="${i#*=}"
                ;;
            --symlink_target_path=*)
                VENV_SYMLINK_TARGET_PATH="${i#*=}"
                ;;
            -f|--venv-force-remove)
                FORCE_REMOVE_VENV="y"
                ;;
            -r|--venv-reuse)
                REUSE_VENV="y"
                ;;
            --system-site-packages)
                VENV_SYSTEM_SITE_PACKAGES_ARGS="--system-site-packages"
                ;;
            --help)
                usage
                ;;
            *)
                print_colored "Unknown option: $i" "ERROR" >&2
                usage
                ;;
        esac
    done

    # Validate PYTHON_VERSION against MIN_PY_VERSION if specified
    # Also handles the case where PYTHON_VERSION is empty, implying MIN_PY_VERSION as the effective target for checks.
    local EFFECTIVE_TARGET_PY_VERSION="${PYTHON_VERSION:-${MIN_PY_VERSION}}"
    local REQ_VER_NUM=$(printf "%02d%02d%02d" $(echo "${EFFECTIVE_TARGET_PY_VERSION}" | tr '.' ' '))
    local MIN_VER_NUM=$(printf "%02d%02d%02d" $(echo "${MIN_PY_VERSION}" | tr '.' ' '))

    if [ "${REQ_VER_NUM}" -lt "${MIN_VER_NUM}" ]; then
        print_colored "Requested Python version (${PYTHON_VERSION:-default}) is lower than the minimum required version (${MIN_PY_VERSION}). Aborting." "ERROR" >&2
        exit 1
    fi

    # Validate symlink_target_path requires venv_path
    if [ -n "${VENV_SYMLINK_TARGET_PATH}" ] && [ -z "${VENV_PATH}" ]; then
        print_colored "--symlink_target_path can only be used when --venv_path is also specified." "ERROR" >&2
        exit 1
    fi

    # Handle --venv-force-remove and --venv-reuse conflicts
    if [ "${FORCE_REMOVE_VENV}" = "y" ] && [ "${REUSE_VENV}" = "y" ]; then
        print_colored "Cannot use both --venv-force-remove and --venv-reuse simultaneously. Please choose one." "ERROR" >&2
        exit 1
    fi

    # Check if venv_path exists and handle based on options
    if [ -n "$VENV_PATH" ]; then # Only proceed if VENV_PATH was provided
        # Also check symlink target path if specified
        local CHECK_PATHS=("${VENV_PATH}")
        if [ -n "${VENV_SYMLINK_TARGET_PATH}" ]; then
            CHECK_PATHS+=("${VENV_SYMLINK_TARGET_PATH}")
        fi
        
        for CHECK_PATH in "${CHECK_PATHS[@]}"; do
            if [ -e "$CHECK_PATH" ]; then # Path exists
                if [ "${FORCE_REMOVE_VENV}" = "y" ]; then
                    echo -e "${TAG_INFO} --venv-force-remove specified. Removing existing path at ${CHECK_PATH}..." >&2
                    if ! rm -rf "${CHECK_PATH}"; then
                        print_colored "Failed to remove existing path at ${CHECK_PATH}. Aborting." "ERROR" >&2
                        exit 1
                    fi
                elif [ "${REUSE_VENV}" = "y" ]; then
                    # For reuse, only check the final venv path (VENV_PATH), not the symlink target
                    if [ "${CHECK_PATH}" = "${VENV_PATH}" ]; then
                        if check_venv_validity "${VENV_PATH}"; then
                            echo -e "${TAG_INFO} --venv-reuse specified and existing virtual environment is valid. Skipping venv creation." >&2
                            SKIP_VENV_CREATION="y"
                        else
                            echo -e "${TAG_WARN} --venv-reuse specified, but existing virtual environment at ${VENV_PATH} is invalid. Attempting to recreate it." >&2
                            # Remove both paths if invalid
                            for REMOVE_PATH in "${CHECK_PATHS[@]}"; do
                                if [ -e "${REMOVE_PATH}" ]; then
                                    if ! rm -rf "${REMOVE_PATH}"; then
                                        print_colored "Failed to remove invalid path at ${REMOVE_PATH}. Aborting." "ERROR" >&2
                                        exit 1
                                    fi
                                fi
                            done
                        fi
                    fi
                else
                    print_colored "Path already exists: ${CHECK_PATH}. Please remove it or choose a different path, or use --venv-force-remove to force recreation, or --venv-reuse to reuse it." "HINT" >&2
                    exit 1
                fi
            fi
        done
    fi

    echo -e "${TAG_INFO} Starting Python installation and environment setup..."
    echo -e "${TAG_INFO} Requested Python Version: ${PYTHON_VERSION:-OS Default/Min (${MIN_PY_VERSION})}"

    # Call install_python_and_dependencies and capture its output (the python executable path)
    local INSTALLED_PYTHON_EXEC
    INSTALLED_PYTHON_EXEC=$(install_python_and_dependencies "${PYTHON_VERSION}" "${OS_ID}" "${OS_VERSION}")
    local INSTALL_PY_STATUS=$? # Capture the exit status of install_python_and_dependencies

    if [ ${INSTALL_PY_STATUS} -ne 0 ]; then
        print_colored "Python and Virtual environment setup failed. Exiting." "ERROR" >&2
        exit 1
    fi

    # Ensure INSTALLED_PYTHON_EXEC is not empty (it would be if installation failed or was skipped due to an error)
    if [ -z "${INSTALLED_PYTHON_EXEC}" ]; then
        print_colored "Could not determine installed Python executable or Python installation failed. Exiting." "ERROR" >&2
        exit 1
    fi

    # Conditionally call setup_venv based on VENV_PATH
    if [ -n "${VENV_PATH}" ]; then
        echo -e "${TAG_INFO} Virtual Environment Path: ${VENV_PATH}"

        setup_venv "${INSTALLED_PYTHON_EXEC}" "${VENV_PATH}" "${SKIP_VENV_CREATION}" "${VENV_SYSTEM_SITE_PACKAGES_ARGS}" "${VENV_SYMLINK_TARGET_PATH}"
        if [ $? -ne 0 ]; then
            print_colored "Virtual environment setup failed. Exiting." "ERROR" >&2
            exit 1
        fi
        echo -e "${TAG_SUCC} Script execution completed successfully."
        if [ -n "${VENV_SYMLINK_TARGET_PATH}" ]; then
            echo -e "${TAG_INFO} Virtual environment created at: ${VENV_SYMLINK_TARGET_PATH}"
            echo -e "${TAG_INFO} Symbolic link created at: ${VENV_PATH}"
        fi
        echo -e "${TAG_INFO} To activate the virtual environment, run:"
        echo -e "${COLOR_BRIGHT_YELLOW_ON_BLACK}  source ${VENV_PATH}/bin/activate ${COLOR_RESET}"
    else
        echo -e "${TAG_INFO} No --venv_path specified. Skipping virtual environment creation."
        echo -e "${TAG_SUCC} Script execution completed successfully (Python installed)."
    fi
}

# Call the main function with all script arguments
main "$@"
