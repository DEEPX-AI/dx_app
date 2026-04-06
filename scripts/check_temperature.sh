#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

# default values
SLEEP_INTERVAL=1        # in seconds
TIMEOUT=-1              # in seconds, -1 for infinite

# color env settings
source "${SCRIPT_DIR}/color_env.sh"
source "${SCRIPT_DIR}/common_util.sh"

# Function to exit with an error message and hint
exit_with_message() {
    local error_message="$1"
    print_colored_v2 "ERROR" "$error_message"
    exit 1
}

# Function to display help message
show_help() {
    echo -e "Usage: ${COLOR_CYAN}$(basename "$0") --wait_target_temp=<temperature> [OPTIONS]${COLOR_RESET}"
    echo -e ""
    echo -e "Required:"
    echo -e "  ${COLOR_GREEN}--wait_target_temp=<temperature>${COLOR_RESET}   Wait until the NPU temperature drops below the specified value (in Celsius) before proceeding"
    echo -e ""
    echo -e "Options:"
    echo -e "  ${COLOR_GREEN}[--sleep_interval=<seconds>]${COLOR_RESET}       Sleep interval in seconds between temperature checks (default: ${SLEEP_INTERVAL})"
    echo -e "  ${COLOR_GREEN}[--timeout=<seconds>]${COLOR_RESET}              Timeout in seconds, -1 for infinite (default: ${TIMEOUT})"
    echo -e "  ${COLOR_GREEN}[-h|--help]${COLOR_RESET}                        Display this help message and exit"
    echo -e ""
    echo -e "${COLOR_BOLD}Examples:${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --wait_target_temp=80${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --wait_target_temp=80 --sleep_interval=30 --timeout=600${COLOR_RESET}"
    echo -e "  ${COLOR_YELLOW}$0 --wait_target_temp=75 --sleep_interval=10${COLOR_RESET}"
    echo -e ""

    if [ "$1" == "error" ] && [[ -z "$2" ]]; then
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

check_npu_temperature() {
    # Get NPU temperature information using dxrt-cli
    local dxrt_output
    dxrt_output=$(dxrt-cli -s 2>/dev/null)

    if [ $? -ne 0 ]; then
        print_colored_v2 "ERROR" "Failed to get NPU temperature information from dxrt-cli"
        echo "-1"
        return 1
    fi

    # Extract all NPU temperatures and find the maximum
    local max_temp=-1
    while IFS= read -r line; do
        if [[ $line =~ NPU\ [0-9]+:.*temperature\ ([0-9]+)\'C ]]; then
            local temp="${BASH_REMATCH[1]}"
            if (( temp > max_temp )); then
                max_temp=$temp
            fi
        fi
    done <<< "$dxrt_output"

    if [ $max_temp -eq -1 ]; then
        print_colored_v2 "ERROR" "No NPU temperature information found"
        echo "-1"
        return 1
    fi

    echo "$max_temp"
    return 0
}

wait_target_temp() {
    local target_temp=$1
    local sleep_interval=${2:-30}
    local timeout=${3:--1}

    print_colored_v2 "INFO" "Waiting for NPU temperature to drop below ${target_temp}°C..."
    print_colored_v2 "INFO" "Sleep interval: ${sleep_interval}s, Timeout: $([ $timeout -eq -1 ] && echo 'infinite' || echo ${timeout}s)"

    local start_time
    start_time=$(date +%s)

    local current_temp
    while true; do
        current_temp=$(check_npu_temperature)
        if [ $? -ne 0 ]; then
            print_colored_v2 "ERROR" "Failed to get NPU temperature"
            return 1
        fi

        if (( current_temp < target_temp )); then
            print_colored_v2 "GREEN" "[OK] Current NPU temperature ${current_temp}°C is below target ${target_temp}°C."
            return 0
        else
            print_colored_v2 "YELLOW" "Current NPU temperature ${current_temp}°C is above target ${target_temp}°C. Waiting..."
        fi

        # Check timeout
        if [ $timeout -ne -1 ]; then
            local current_time
            current_time=$(date +%s)
            local elapsed=$((current_time - start_time))
            if (( elapsed >= timeout )); then
                print_colored_v2 "ERROR" "Timeout reached (${timeout}s). Current temperature: ${current_temp}°C"
                return 5
            fi
        fi

        sleep $sleep_interval
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait_target_temp=*)
            TARGET_TEMP="${1#*=}"
            ;;
        --sleep_interval=*)
            SLEEP_INTERVAL="${1#*=}"
            ;;
        --timeout=*)
            TIMEOUT="${1#*=}"
            ;;
        -h|--help)
            show_help
            ;;
        *)
            print_colored "Unknown option: $1"
            show_help "error"
            ;;
    esac
    shift
done

# usage
if [ -z "$TARGET_TEMP" ]; then
    exit_with_message "TARGET_TEMP(${TARGET_TEMP}) does not exist."
fi

main() {
    if [ -n "$TARGET_TEMP" ]; then
        wait_target_temp "$TARGET_TEMP" "$SLEEP_INTERVAL" "$TIMEOUT"
        return $?
    fi
}

main
exit $?
