# DX-APP YOLO Customizing Guide

This guide explains how to add or adapt YOLO-family models in DX-APP. It is intended for contributors who need to onboard a new YOLO model, select the correct postprocessor, tune thresholds, or adjust factory code after model conversion.

DX-APP supports YOLO examples through a common factory pattern:

1. choose the closest existing YOLO postprocessor family,
2. generate or copy the example source tree,
3. tune `config.json`,
4. build the target,
5. run image/video verification,
6. update registry/test metadata when the model becomes a maintained example.

---

## 1. Choose the Right YOLO Family

Start by identifying the output format of the compiled `.dxnn` model. The model name alone is helpful, but the output tensor layout is the final source of truth.

| Model type | Typical output | `add_model.sh --postprocessor` | C++ postprocessor | Python postprocessor |
|------------|----------------|--------------------------------|-------------------|----------------------|
| YOLOv3/v4/v5-style anchor-based detection | box + objectness + class scores | `yolov5` | `YOLOv5Postprocessor` | `YOLOv5Postprocessor` |
| YOLOv7 anchor-based detection | YOLOv7 anchors/strides | `yolov7` | `YOLOv7Postprocessor` | `YOLOv5Postprocessor` with model config |
| YOLOX | anchor-free YOLOX-style detection | `yolox` | `YOLOXPostprocessor` | `YOLOXPostprocessor` |
| YOLOv8/v9/v10/v11/v12/v26 detection | anchor-free / DFL-style detection, no objectness field | `yolov8`, `yolov9`, `yolov10`, `yolov11`, `yolov12`, `yolov26` | matching `YOLOv*Postprocessor` | `YOLOv8Postprocessor` family |
| YOLO segmentation | boxes + masks/prototypes | `yolov8seg`, `yolov26seg`, `yolov5seg` | task-specific segmentation postprocessor | task-specific segmentation postprocessor |
| YOLO pose | boxes + keypoints | `yolov5pose`, `yolov8pose`, `yolov26pose` | task-specific pose postprocessor | task-specific pose postprocessor |
| YOLO face detection | face boxes and optional landmarks | `yolov5face`, `yolov7face` | face-specific postprocessor | face-specific postprocessor |
| YOLO OBB | oriented boxes | `yolov26obb` | OBB postprocessor | OBB postprocessor |
| PPU YOLO variants | hardware-assisted postprocess output | `yolov5_ppu`, `yolov7_ppu`, `yolov8_ppu`, etc. | PPU postprocessor | PPU postprocessor |

!!! tip
    If the model is a new detection-only YOLO variant and its output is compatible with the YOLOv8+ anchor-free/DFL family, start with `--postprocessor yolov8`. If it has explicit objectness and anchor-grid decoding, start with `--postprocessor yolov5` or `--postprocessor yolov7`.

---

## 2. Generate a New YOLO Example

Use `scripts/add_model.sh` to create the C++ and Python example trees from the closest reference model.

```bash
# YOLOv8-style detection model
./scripts/add_model.sh yolov30 object_detection --postprocessor yolov8 --lang both

# YOLOv7-style anchor-based model
./scripts/add_model.sh yolov7_w6 object_detection --base-model yolov7 --postprocessor yolov7 --lang both

# YOLOX-style model
./scripts/add_model.sh custom_yolox object_detection --postprocessor yolox --lang both
```

Generated files follow the standard layout:

```text
src/cpp_example/object_detection/<model_name>/
├── config.json
├── factory/<model_name>_factory.hpp
├── <model_name>_sync.cpp
└── <model_name>_async.cpp

src/python_example/object_detection/<model_name>/
├── config.json
├── factory/<model_name>_factory.py
├── <model_name>_sync.py
├── <model_name>_async.py
├── <model_name>_sync_cpp_postprocess.py
└── <model_name>_async_cpp_postprocess.py
```

After generation, review both language trees. Do not assume a generated factory is final; the selected template only gives a safe starting point.

---

## 3. Tune `config.json`

Most YOLO customization is done through each example's `config.json`. Keep C++ and Python configs aligned unless a language-specific path intentionally differs.

Common keys:

| Key | Meaning | Typical use |
|-----|---------|-------------|
| `obj_threshold` | objectness threshold | YOLOv5/v7-style models with objectness |
| `score_threshold` or `conf_threshold` | class confidence threshold | all detection models |
| `nms_threshold` | IoU threshold for NMS | all detection models |
| `num_classes` | number of classes | custom datasets that are not COCO-80 |
| `class_names` | display labels | custom dataset labels |
| `anchors` | anchor map by stride | custom anchor-based YOLOv5/v7 variants |
| `strides` | output strides | non-standard multi-scale layouts |

Example for a custom 3-class YOLOv5-style model:

```json
{
  "obj_threshold": 0.25,
  "score_threshold": 0.30,
  "nms_threshold": 0.45,
  "num_classes": 3,
  "class_names": ["person", "vehicle", "animal"],
  "anchors": {
    "8": [[10, 13], [16, 30], [33, 23]],
    "16": [[30, 61], [62, 45], [59, 119]],
    "32": [[116, 90], [156, 198], [373, 326]]
  },
  "strides": [8, 16, 32]
}
```

Example for a custom YOLOv8-style model:

```json
{
  "score_threshold": 0.30,
  "nms_threshold": 0.45,
  "num_classes": 3,
  "class_names": ["person", "vehicle", "animal"]
}
```

!!! warning
    `num_classes` must match the compiled model output. If the model was compiled with a different class count, changing only `config.json` will not fix the tensor shape.

---

## 4. Review the Factory Code

Factories assemble the preprocessor, postprocessor, visualizer, and model metadata. For YOLO detection models they normally use `LetterboxPreprocessor`/`DetectionPreprocessor`, a YOLO postprocessor, and `DetectionVisualizer`.

C++ YOLOv8-style factory pattern:

```cpp
PostprocessorPtr<DetectionResult> createPostprocessor(
    int input_width, int input_height, bool is_ort_configured = false) override {
    return std::make_unique<YOLOv8Postprocessor>(
        input_width, input_height,
        score_threshold_, nms_threshold_,
        is_ort_configured,
        num_classes_,
        class_names_
    );
}

void loadConfig(const dxapp::ModelConfig& config) override {
    score_threshold_ = config.get<float>("score_threshold", score_threshold_);
    nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    class_names_ = config.get_string_list("class_names");
    num_classes_ = config.get<int>("num_classes", num_classes_);
}
```

Python YOLOv8-style factory pattern:

```python
def create_postprocessor(self, input_width: int, input_height: int):
    return YOLOv8Postprocessor(input_width, input_height, self.config)
```

Check these points after generation:

- `get_model_name()` returns the new model name.
- `get_task_type()` remains `object_detection` for detection models.
- C++ and Python use the same postprocessor family.
- Thresholds and class metadata are loaded from `config.json`.
- Specialized tasks use the correct task-specific visualizer and runner.

---

## 5. Understand Output Shape Compatibility

YOLO postprocessors support common DX-APP output patterns, but a converted model may still differ.

Typical YOLOv5/v7-style outputs:

- single decoded tensor: `[1, N, 5 + num_classes]`
- multi-scale NPU tensors: `[1, C, H, W]`, where `C = anchors * (5 + num_classes)`
- objectness is present and combined with class scores

Typical YOLOv8+ detection outputs:

- anchor-free detection tensors
- no separate objectness field
- postprocessor uses class score and NMS thresholds

If verification fails due to unexpected tensor shape:

1. run with `--show-log` if the example supports verbose logs,
2. enable tensor dumping with `--dump-tensors`,
3. inspect output tensor names, dimensions, and value ranges,
4. compare with the selected reference model,
5. adjust the factory/config only if the output family is compatible,
6. implement a focused postprocessor variant only when the tensor contract is genuinely different.

---

## 6. Build and Run

Build the generated targets:

```bash
./build.sh --type debug --target <model_name>_sync
./build.sh --type debug --target <model_name>_async
```

Run C++ examples:

```bash
./bin/<model_name>_sync  -m assets/models/<Model>.dxnn -i sample/img/sample_kitchen.jpg --no-display -l 1
./bin/<model_name>_async -m assets/models/<Model>.dxnn -v assets/videos/dance-group.mov --no-display -l 1
```

Run Python examples:

```bash
python src/python_example/object_detection/<model_name>/<model_name>_sync.py \
  --model assets/models/<Model>.dxnn \
  --image sample/img/sample_kitchen.jpg \
  --no-display \
  --loop 1

python src/python_example/object_detection/<model_name>/<model_name>_async.py \
  --model assets/models/<Model>.dxnn \
  --video assets/videos/dance-group.mov \
  --no-display \
  --loop 1
```

Use saved output for visual checks:

```bash
./bin/<model_name>_sync -m assets/models/<Model>.dxnn -i sample/img/sample_kitchen.jpg --save
```

---

## 7. Verify Before Registry Integration

Before registering a new YOLO model as a maintained example, verify it in both languages.

Recommended sequence:

```bash
# Structure validation
./scripts/dx_tool.sh validate

# Targeted build
./build.sh --type debug --target <model_name>_sync
./build.sh --type debug --target <model_name>_async

# CLI smoke tests
./scripts/dx_tool.sh run --lang cpp --model <model_name>
./scripts/dx_tool.sh run --lang py --model <model_name>

# Broader checks when assets are available
./run_tc.sh --cpp --cli
./run_tc.sh --python
```

For numerical validation, add or update rules only after the visual and CLI checks are stable. Avoid checking in model metadata that points to missing `.dxnn` assets.

---

## 8. Update Registry and Test Metadata

When the model is intended to ship as a first-class example, update the repository metadata together:

- `config/model_registry.json`
- `config/test_models.conf`
- C++ example directory
- Python example directory
- verification rules if numerical checks are required
- documentation or release notes if this model is user-facing

A registry entry should identify:

- model name,
- `.dxnn` file name,
- task/category,
- postprocessor family,
- input size,
- support status,
- any required config metadata.

Do not add registry entries for local experiments or customer-only artifacts unless the release explicitly includes them.

---

## 9. Troubleshooting

### No detections appear

- Lower `score_threshold`/`conf_threshold` temporarily.
- Check whether the selected postprocessor family matches the output tensor format.
- Confirm `num_classes` matches the model output.
- Confirm preprocessing uses letterbox behavior expected by the model.

### Boxes are shifted or scaled incorrectly

- Check model input width/height.
- Check whether the model expects letterbox preprocessing or direct resize.
- Compare C++ and Python outputs on the same image.
- Verify box scaling uses the `PreprocessContext` from the active preprocessor.

### NMS removes too many boxes

- Increase or decrease `nms_threshold` depending on overlap behavior.
- Lower `score_threshold` to check whether candidate boxes exist before NMS.

### C++ and Python results differ

- Ensure both configs contain the same thresholds, class count, and labels.
- Ensure both factories use the same postprocessor family.
- Run one image with display disabled and compare saved output or serialized verification output.
- Check whether one path uses a C++ binding variant while the other uses pure Python postprocess.

### Build target is missing

- Confirm the generated directory name matches `<model_name>`.
- Run `./scripts/dx_tool.sh validate` to catch layout issues.
- Check that CMake discovered the generated example directory.

---

## 10. Release Checklist

Before merging a YOLO customization:

- [ ] The selected postprocessor family matches the model output tensor contract.
- [ ] C++ and Python factories use matching preprocessing/postprocessing behavior.
- [ ] `config.json` has correct thresholds and `num_classes`.
- [ ] The model runs on at least one image input.
- [ ] Async execution runs on image or video input as appropriate for the task.
- [ ] Saved visualization looks correct.
- [ ] `dx_tool.sh validate` passes.
- [ ] Targeted build passes for sync and async targets.
- [ ] Registry/test metadata is updated only for release-supported models.
- [ ] Documentation or release notes are updated for user-facing changes.
