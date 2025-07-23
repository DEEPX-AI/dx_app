#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_APP_PATH=$(realpath -s "${SCRIPT_DIR}/..")

# color env settings
source "${SCRIPT_DIR}/color_env.sh"
source "${SCRIPT_DIR}/common_util.sh"

pushd $DX_APP_PATH

dxapp_name="run_detector"
print_colored "DX_APP_PATH: $DX_APP_PATH" "INFO"

if ! test -e $DX_APP_PATH/bin/$dxapp_name; then
    print_colored "dx_app is not built. Building dx_app first before running the example." "INFO"
    ./build.sh --clean
fi

check_valid_dir_or_symlink() {
    local path="$1"
    if [ -d "$path" ] || { [ -L "$path" ] && [ -d "$(readlink -f "$path")" ]; }; then
        return 0
    else
        return 1
    fi
}

if [ check_valid_dir_or_symlink "./assets" ]; then
    print_colored "Assets directory already exists. Skipping download." "INFO"
else
    print_colored "Assets not found. Downloading now via setup.sh..." "INFO"
    ./setup.sh
fi

$DX_APP_PATH/bin/$dxapp_name -c $DX_APP_PATH/example/run_detector/yolov5s3_example.json

popd
