#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
pushd $SCRIPT_DIR
FORCE_ARGS=""

# Function to display help message
show_help() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Options:"
  echo "  --docker_volume_path=<path>    Set Docker volume path (required in container mode)"
  echo "  [--force]                      Force overwrite if the file already exists"
  echo "  [--help]                       Show this help message"

  if [ "$1" == "error" ]; then
    echo "Error: Invalid or missing arguments."
    exit 1
  fi
  exit 0
}

# Default values
DOCKER_VOLUME_PATH=${DOCKER_VOLUME_PATH}
echo "=== DOCKER_VOLUME_PATH($DOCKER_VOLUME_PATH) is set ==="

# Parse arguments
for i in "$@"; do
  case $i in
    --docker_volume_path=*)
      DOCKER_VOLUME_PATH="${i#*=}"
      ;;
    --force)
      FORCE_ARGS="--force"
      ;;
    --help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      show_help "error"
      ;;
  esac
  shift
done

MODEL_PATH=./assets/models
VIDEO_PATH=./assets/videos
CONTAINER_MODE=false

# Check if running in a container
if grep -qE "/docker|/lxc|/containerd" /proc/1/cgroup || [ -f /.dockerenv ]; then
    CONTAINER_MODE=true
    echo "(container mode detected)"
    
    if [ -z "$DOCKER_VOLUME_PATH" ]; then
        echo "Error: --docker_volume_path must be provided in container mode."
        show_help "error"
        exit 1
    fi

    SETUP_MODEL_ARGS="--output=${MODEL_PATH} --symlink_target_path=${DOCKER_VOLUME_PATH}/res/models"
    SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=${DOCKER_VOLUME_PATH}/res/videos"
else
    echo "(host mode detected)"
    SETUP_MODEL_ARGS="--output=${MODEL_PATH} --symlink_target_path=../workspace/res/models"
    SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=../workspace/res/videos"
fi

echo "MODEL_PATH: ${MODEL_PATH}"
MODEL_REAL_PATH=$(readlink -f "$MODEL_PATH")
# Check and set up models
if [ ! -d "$MODEL_REAL_PATH" ] || [ "$FORCE_ARGS" != "" ]; then
  echo "models directory not found. Running setup models script... ($MODEL_REAL_PATH)"
  ./setup_sample_models.sh $SETUP_MODEL_ARGS $FORCE_ARGS || { echo "Setup models script failed."; rm -rf $MODEL_PATH; exit 1; }
else
  echo "models directory found. ($MODEL_REAL_PATH)"
fi

echo "VIDEO_PATH: ${VIDEO_PATH}"
VIDEO_REAL_PATH=$(readlink -f "$VIDEO_PATH")
# Check and set up models
if [ ! -d "$VIDEO_REAL_PATH" ] || [ "$FORCE_ARGS" != "" ]; then
  echo "Video directory not found. Running setup models script... ($VIDEO_REAL_PATH)"
  ./setup_sample_videos.sh $SETUP_VIDEO_ARGS $FORCE_ARGS || { echo "Setup models script failed."; rm -rf $VIDEO_PATH; exit 1; }
else
  echo "Video directory found. ($VIDEO_REAL_PATH)"
fi

popd
