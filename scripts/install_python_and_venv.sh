#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

# color env settings
source "${SCRIPT_DIR}/color_env.sh"

# Global variables for script configuration
MIN_PY_VERSION="3.8.10"

# ---
## Usage Information
# ---
usage() {
    echo -e "Usage: ${COLOR_CYAN}$0 [OPTIONS]${COLOR_RESET}"
    echo -e "Install a specified Python version and optionally set up a virtual environment."
    echo -e ""
    echo -e "${COLOR_BOLD}Options:${COLOR_RESET}"
    echo -e "  ${COLOR_GREEN}--python_version=<VERSION>${COLOR_RESET}  Specify the Python version to install (e.g., 3.10.4)."
    echo -e "                            Minimum supported version: ${MIN_PY_VERSION}."
    echo -e "                            If not specified:"
    echo -e "                              - For Ubuntu 20.04+, the OS default Python 3 will be used."
    echo -e "                              - For Ubuntu 18.04, Python ${MIN_PY_VERSION} will be source-built."
    echo -e "  ${COLOR_GREEN}--venv_path=<PATH>${COLOR_RESET}          Specify the path for the virtual environment."
    echo -e "                            If this option is omitted, no virtual environment will be created."
    echo -e "  ${COLOR_GREEN}--help${COLOR_RESET}                      Display this help message and exit."
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --python_version=3.10.4 --venv_path=/opt/my_venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --python_version=3.9.18  # Installs Python, but no venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --venv_path=/usr/local/py_env # Installs default Python, creates venv${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 # Installs default Python, but no venv${COLOR_RESET}"
    echo -e ""
    exit 0
}

# ---
## Check if Python version is already installed and usable, or if a higher suitable version exists.
# Returns 0 if installed and usable, 1 otherwise.
# Outputs the executable path to stdout if found and usable.
# Logs to stderr.
# Arguments:
#   $1: UBUNTU_VERSION - Current Ubuntu release version (e.g., "20.04")
#   $2: REQUESTED_PY_VERSION (optional) - The specific Python version requested (e.g., "3.8.2").
#                                         If empty, means OS default/MIN_PY_VERSION is implied.
#   $3: MIN_REQUIRED_PY_VERSION - The absolute minimum Python version required by the script.
# ---
is_python_installed() {
    local UBUNTU_VERSION="${1}"
    local REQUESTED_PY_VERSION="${2}"
    local MIN_REQUIRED_PY_VERSION="${3}"
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

        # Try apt path first for newer Ubuntus, or if command exists
        if { [ "$UBUNTU_VERSION" = "24.04" ] || [ "$UBUNTU_VERSION" = "22.04" ] || [ "$UBUNTU_VERSION" = "20.04" ]; } || command -v "${cmd}" &>/dev/null; then
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
#         If empty, the OS default Python 3 version will be installed for Ubuntu 20.04+.
#         For Ubuntu 18.04, MIN_PY_VERSION will be used if TARGET_INSTALL_PY_VERSION is empty.
#   $2: UBUNTU_VERSION - The current Ubuntu release version.
install_python_and_dependencies() {
    local TARGET_INSTALL_PY_VERSION="${1}"
    local UBUNTU_VERSION="${2}"

    # Temporarily disable 'set -e' for controlled error handling and 'set -x' for cleaner output capture
    local OPT_E_STATE="$-"
    local OPT_X_STATE="$-"
    case "${OPT_E_STATE}" in *e*) set +e;; esac
    case "${OPT_X_STATE}" in *x*) set +x;; esac

    exec 3>&1 # Save stdout to fd 3
    exec >&2  # Redirect stdout to stderr

    echo -e "${TAG_INFO} Starting Python installation/dependency checks for ${TARGET_INSTALL_PY_VERSION:-default}..."

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
        PYTHON_EXE_NAME="python3" # Default for Ubuntu 20.04+
    fi

    # Check if Python is already installed OR if a higher suitable version exists
    local IS_INSTALLED_RESULT
    IS_INSTALLED_RESULT=$(is_python_installed "${UBUNTU_VERSION}" "${TARGET_INSTALL_PY_VERSION}" "${MIN_PY_VERSION}")
    
    if [ -n "${IS_INSTALLED_RESULT}" ]; then
        DX_PYTHON_EXEC_OUT="${IS_INSTALLED_RESULT}"
        echo -e "${TAG_SKIP} A suitable Python installation is already present (${DX_PYTHON_EXEC_OUT})."
        echo -e "${TAG_INFO} Ensuring required development and venv packages for ${DX_PYTHON_EXEC_OUT} are installed."
        
        # Install dev/venv packages even if main interpreter is skipped
        if [ "$UBUNTU_VERSION" = "24.04" ] || [ "$UBUNTU_VERSION" = "22.04" ] || [ "$UBUNTU_VERSION" = "20.04" ]; then
            local TARGET_MAJOR_MINOR=$(echo "${DX_PYTHON_EXEC_OUT}" | sed -n 's/.*python\([0-9]\.[0-9]\+\).*/\1/p')
            if [ -z "${TARGET_MAJOR_MINOR}" ]; then TARGET_MAJOR_MINOR="3"; fi # Fallback for 'python3'

            # Check if dev and venv packages are already installed
            if dpkg -s "python${TARGET_MAJOR_MINOR}-dev" >/dev/null 2>&1 && dpkg -s "python${TARGET_MAJOR_MINOR}-venv" >/dev/null 2>&1; then
                echo -e "${TAG_SKIP} Python${TARGET_MAJOR_MINOR}-dev and python${TARGET_MAJOR_MINOR}-venv are already installed. Skipping."
            else
                if ! sudo apt-get update; then
                    echo -e "${TAG_ERROR} Failed to update apt repositories"
                    INSTALL_STATUS=1
                elif ! sudo apt install -y gnupg gpg-agent; then
                    echo -e "${TAG_ERROR} Failed to install gnupg gpg-agent."
                    INSTALL_STATUS=1
                elif ! sudo add-apt-repository -y ppa:deadsnakes/ppa; then
                    echo -e "${TAG_ERROR} Failed to add deadsnakes PPA."
                    INSTALL_STATUS=1
                elif ! sudo apt-get update; then
                    echo -e "${TAG_ERROR} Failed to update apt repositories after adding PPA."
                    INSTALL_STATUS=1
                elif ! sudo apt-get install -y python${TARGET_MAJOR_MINOR}-dev python${TARGET_MAJOR_MINOR}-venv; then
                    echo -e "${TAG_ERROR} Failed to install python${TARGET_MAJOR_MINOR}-dev and/or python${TARGET_MAJOR_MINOR}-venv."
                    INSTALL_STATUS=1
                fi
            fi
        fi
        # Source-built Python usually includes venv, dev headers etc. if built correctly.
        # No extra apt installs needed here for 18.04 source builds.
    else
        echo -e "${TAG_INFO} No suitable Python installation found. Proceeding with requested installation."

        if [ "$UBUNTU_VERSION" = "24.04" ] || [ "$UBUNTU_VERSION" = "22.04" ] || [ "$UBUNTU_VERSION" = "20.04" ]; then
            if [ -n "${TARGET_INSTALL_PY_VERSION}" ]; then
                echo -e "${TAG_INFO} Installing python ${PY_MAJOR_MINOR} version using apt..."
                if ! sudo apt-get update; then
                    echo -e "${TAG_ERROR} Failed to update apt repositories"
                    INSTALL_STATUS=1
                elif ! sudo apt install -y gnupg gpg-agent; then
                    echo -e "${TAG_ERROR} Failed to install gnupg gpg-agent."
                    INSTALL_STATUS=1
                elif ! sudo add-apt-repository -y ppa:deadsnakes/ppa; then
                    echo -e "${TAG_ERROR} Failed to add deadsnakes PPA."
                    INSTALL_STATUS=1
                elif ! sudo apt-get update; then
                    echo -e "${TAG_ERROR} Failed to update apt repositories after adding PPA."
                    INSTALL_STATUS=1
                elif ! sudo apt-get install -y python${PY_MAJOR_MINOR} python${PY_MAJOR_MINOR}-dev python${PY_MAJOR_MINOR}-venv; then
                    echo -e "${TAG_ERROR} apt installation failed for python${PY_MAJOR_MINOR}."
                    INSTALL_STATUS=1
                fi
                DX_PYTHON_EXEC_OUT="python${PY_MAJOR_MINOR}"
            else
                echo -e "${TAG_INFO} Installing OS default python3 version and dependencies using apt..."
                if ! sudo apt-get update; then
                    echo -e "${TAG_ERROR} Failed to update apt repositories"
                    INSTALL_STATUS=1
                elif ! sudo apt-get install -y python3 python3-dev python3-venv; then
                    echo -e "${TAG_ERROR} apt installation failed for python3."
                    INSTALL_STATUS=1
                fi
                DX_PYTHON_EXEC_OUT="python3"
            fi
        elif [ "$UBUNTU_VERSION" = "18.04" ]; then
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

                if ! wget --no-check-certificate https://www.python.org/ftp/python/${TARGET_INSTALL_PY_VERSION}/Python-${TARGET_INSTALL_PY_VERSION}.tgz || \
                    ! tar xvf Python-${TARGET_INSTALL_PY_VERSION}.tgz || \
                    ! (cd Python-${TARGET_INSTALL_PY_VERSION} && ./configure --enable-optimizations && make -j$(nproc) && sudo make altinstall); then
                    echo -e "${TAG_ERROR} Source build for Python ${TARGET_INSTALL_PY_VERSION} failed."
                    INSTALL_STATUS=1
                fi
                popd >/dev/null # Pop back to original directory
                rm -rf "${BUILD_DIR}" # Clean up build directory

                if [ ${INSTALL_STATUS} -eq 0 ]; then
                    DX_PYTHON_EXEC_OUT="python${PY_MAJOR_MINOR}" # Set executable for source build
                fi
            fi
        else
            echo -e "${TAG_ERROR} Unsupported Ubuntu version: $UBUNTU_VERSION"
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
## Setup Virtual Environment
# ---
# Arguments:
#   $1: DX_PYTHON_EXEC - The command to execute the installed Python (e.g., python3.10).
#   $2: VENV_PATH - The desired path for the virtual environment.
setup_venv() {
    local DX_PYTHON_EXEC="${1}"
    local VENV_PATH="${2}"

    # Temporarily disable 'set -x' for cleaner output during venv setup steps
    local OPT_X_STATE="$-"
    case "${OPT_X_STATE}" in *x*) set +x;; esac

    if [ -z "${DX_PYTHON_EXEC}" ]; then
        echo -e "${TAG_ERROR} Python executable not provided to setup_venv." >&2
        return 1
    fi
    if [ -z "${VENV_PATH}" ]; then
        echo -e "${TAG_ERROR} Virtual environment path not provided to setup_venv." >&2
        return 1
    fi

    echo -e "${TAG_INFO} Setting up Virtual Environment at ${VENV_PATH} using ${DX_PYTHON_EXEC}..."
    if ! "${DX_PYTHON_EXEC}" -m venv "${VENV_PATH}"; then
        echo -e "${TAG_ERROR} Failed to create virtual environment at ${VENV_PATH}." >&2
        case "${OPT_X_STATE}" in *x*) set -x;; esac # Restore set -x before returning
        return 1
    fi

    # Activate the venv temporarily for pip operations
    echo -e "${TAG_INFO} Activating virtual environment for package upgrades..."
    if ! source "${VENV_PATH}/bin/activate"; then # Use 'source' or '.' here
        echo -e "${TAG_ERROR} Failed to activate virtual environment." >&2
        case "${OPT_X_STATE}" in *x*) set -x;; esac # Restore set -x before returning
        return 1
    fi

    echo -e "${TAG_INFO} Upgrading pip, wheel, and setuptools..."
    local UBUNTU_VERSION=$(lsb_release -rs)
    echo -e "${TAG_INFO} *** UBUNTU_VERSION(${UBUNTU_VERSION}) ***"

    local PIP_INSTALL_STATUS=0
    if [ "$UBUNTU_VERSION" = "24.04" ]; then
      if ! pip install --upgrade setuptools; then PIP_INSTALL_STATUS=1; fi
    elif [ "$UBUNTU_VERSION" = "22.04" ] || [ "$UBUNTU_VERSION" = "20.04" ] || [ "$UBUNTU_VERSION" = "18.04" ]; then
      if ! pip install --upgrade pip wheel setuptools; then PIP_INSTALL_STATUS=1; fi
    else
      echo -e "${TAG_WARN} Unsupported Ubuntu version for specific pip upgrade rules: ${UBUNTU_VERSION}" >&2
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
    local VENV_PATH="" # Initialize as empty string
    local UBUNTU_VERSION=$(lsb_release -rs) # Get Ubuntu version once

    # Parse command-line arguments
    for i in "$@"; do
        case $i in
            --python_version=*)
                PYTHON_VERSION="${i#*=}"
                shift # past argument=value
                ;;
            --venv_path=*)
                VENV_PATH="${i#*=}"
                shift # past argument=value
                ;;
            --help)
                usage
                ;;
            *)
                echo -e "${TAG_ERROR} Unknown option: $i" >&2
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
        echo -e "${TAG_ERROR} Requested Python version (${PYTHON_VERSION:-default}) is lower than the minimum required version (${MIN_PY_VERSION}). Aborting." >&2
        exit 1
    fi

    # Check if venv_path exists only if it's provided
    if [ -n "$VENV_PATH" ] && [ -e "$VENV_PATH" ]; then
        echo -e "${TAG_ERROR} Virtual environment path already exists: ${VENV_PATH}. Please remove it or choose a different path." >&2
        exit 1
    fi

    echo -e "${TAG_INFO} Starting Python installation and environment setup..."
    echo -e "${TAG_INFO} Requested Python Version: ${PYTHON_VERSION:-OS Default/Min (${MIN_PY_VERSION})}"

    # Call install_python_and_dependencies and capture its output (the python executable path)
    local INSTALLED_PYTHON_EXEC
    INSTALLED_PYTHON_EXEC=$(install_python_and_dependencies "${PYTHON_VERSION}" "${UBUNTU_VERSION}")
    local INSTALL_PY_STATUS=$? # Capture the exit status of install_python_and_dependencies

    if [ ${INSTALL_PY_STATUS} -ne 0 ]; then
        echo -e "${TAG_ERROR} Python and Virual environment setup failed. Exiting." >&2
        exit 1
    fi

    # Ensure INSTALLED_PYTHON_EXEC is not empty (it would be if installation failed or was skipped due to an error)
    if [ -z "${INSTALLED_PYTHON_EXEC}" ]; then
        echo -e "${TAG_ERROR} Could not determine installed Python executable or Python installation failed. Exiting." >&2
        exit 1
    fi

    # Conditionally call setup_venv based on VENV_PATH
    if [ -n "${VENV_PATH}" ]; then
        echo -e "${TAG_INFO} Virtual Environment Path: ${VENV_PATH}"
        setup_venv "${INSTALLED_PYTHON_EXEC}" "${VENV_PATH}"
        if [ $? -ne 0 ]; then
            echo -e "${TAG_ERROR} Virtual environment setup failed. Exiting." >&2
            exit 1
        fi
        echo -e "${TAG_SUCC} Script execution completed successfully."
        echo -e "${TAG_INFO} To activate the virtual environment, run:"
        echo -e "${COLOR_BRIGHT_YELLOW_ON_BLACK}  source ${VENV_PATH}/bin/activate ${COLOR_RESET}"
    else
        echo -e "${TAG_INFO} No --venv_path specified. Skipping virtual environment creation."
        echo -e "${TAG_SUCC} Script execution completed successfully (Python installed)."
    fi
}

# Call the main function with all script arguments
main "$@"