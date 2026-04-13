#!/usr/bin/env bash
# batch_add_models.sh — Generate C++/Python example dirs for 133 models
# that have supported postprocessors in add_model.sh.
#
# Usage: ./scripts/batch_add_models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DX_APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$DX_APP_ROOT"

# Generate the list of (model_name, task_type, postprocessor) using registry
MODELS=$(python3 << 'PYEOF'
import json, os

with open("config/model_registry.json") as f:
    reg = json.load(f)

cpp_base = "src/cpp_example"
cpp_examples = set()
for cat in os.listdir(cpp_base):
    cat_path = os.path.join(cpp_base, cat)
    if os.path.isdir(cat_path) and cat != "common":
        for d in os.listdir(cat_path):
            if os.path.isdir(os.path.join(cat_path, d)):
                cpp_examples.add(d)

supported_pp = {
    "yolov5", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov12",
    "yolov26", "yolox", "ssd", "nanodet", "damoyolo", "centernet",
    "scrfd", "yolov5face", "yolov7face", "retinaface",
    "yolov5pose", "yolov8pose", "yolov26pose", "yolov26obb", "obb",
    "deeplabv3", "bisenetv1", "bisenetv2", "segformer",
    "yolov8seg", "yolov5seg", "yolact", "yolov26seg",
    "espcn", "zero_dce", "arcface", "clip_image", "clip_text",
    "hand_landmark", "dncnn", "fastdepth",
    "yolov5_ppu", "yolov7_ppu", "yolov5pose_ppu", "scrfd_ppu",
    "efficientnet",
}

TASK_MAP = {
    "classification": "classification",
    "object_detection": "detection",
    "face_detection": "face_detection",
    "pose_estimation": "pose",
    "instance_segmentation": "instance_segmentation",
    "semantic_segmentation": "semantic_segmentation",
    "depth_estimation": "depth_estimation",
    "image_denoising": "image_denoising",
    "super_resolution": "super_resolution",
    "image_enhancement": "image_enhancement",
    "embedding": "embedding",
    "ppu": "ppu",
    "hand_landmark": "hand_landmark",
    "obb_detection": "obb",
    "face_alignment": "face_alignment",
}

can_gen = [m for m in reg if m["model_name"] not in cpp_examples and m["postprocessor"] in supported_pp]
for m in sorted(can_gen, key=lambda x: (x["add_model_task"], x["model_name"])):
    task = TASK_MAP.get(m["add_model_task"], m["add_model_task"])
    print(f'{m["model_name"]}\t{task}\t{m["postprocessor"]}')
PYEOF
)

if [ -z "$MODELS" ]; then
    echo "No models to generate."
    exit 0
fi

total=$(echo "$MODELS" | wc -l)
echo "=== Batch generating $total model examples ==="

success=0
fail=0
skip=0

while IFS=$'\t' read -r model_name task_type postprocessor; do
    [ -z "$model_name" ] && continue

    # Skip if already exists
    if find src/cpp_example -maxdepth 2 -name "$model_name" -type d 2>/dev/null | grep -q .; then
        echo "[SKIP] $model_name (already exists)"
        skip=$((skip + 1))
        continue
    fi

    echo -n "[GEN] $model_name ($task_type, $postprocessor)... "
    if ./scripts/add_model.sh "$model_name" "$task_type" --postprocessor "$postprocessor" --lang both > /tmp/add_model_last.log 2>&1; then
        echo "OK"
        success=$((success + 1))
    else
        echo "FAIL"
        fail=$((fail + 1))
        echo "  Last 5 lines of log:"
        tail -5 /tmp/add_model_last.log | sed 's/^/    /'
    fi
done <<< "$MODELS"

echo ""
echo "=== Batch complete ==="
echo "  Success: $success"
echo "  Failed:  $fail"
echo "  Skipped: $skip"
echo "  Total:   $total"
