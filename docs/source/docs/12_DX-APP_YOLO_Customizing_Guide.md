# DX-APP YOLO Customizing Guide

This guide explains how to add or adapt YOLO-family models in DX-APP. It is intended for contributors who need to onboard a new YOLO model, select the correct postprocessor, tune thresholds, or adjust factory code after model conversion.

DX-APP supports YOLO examples through a common factory pattern. Follow these workflow steps:

- **Step 1**: Choose the Right YOLO Family
- **Step 2**: Generate a New YOLO Example
- **Step 3**: Tune `config.json`
- **Step 4**: Review the Factory Code
- **Step 5**: Understand Output Shape Compatibility
- **Step 6**: Build and Run
- **Step 7**: Verify Before Registry Integration
- **Step 8**: Update Registry and Test Metadata

---

## [Step 1] Choose the Right YOLO Family

Start by identifying the output format of the compiled `.dxnn` model. The model name alone is helpful, but the output tensor layout is the final source of truth.

| Model type | Typical output | `add_model.sh --postprocessor` | C++ postprocessor | Python postprocessor |
|------------|----------------|--------------------------------|-------------------|----------------------|
| YOLOv3/v4/v5-style anchor-based detection | box + objectness + class scores | `yolov5` | `YOLOv5Postprocessor` (the registered `yolov4-*` C++ examples use `EfficientDetPostprocessor`) | `YOLOv5Postprocessor` |
| YOLOv7 anchor-based detection | YOLOv7 anchors/strides | `yolov7` | `YOLOv7Postprocessor` | `YOLOv5Postprocessor` with model config |
| YOLOX | anchor-free YOLOX-style detection | `yolox` | `YOLOXPostprocessor` | `YOLOXPostprocessor` |
| YOLOv8/v9/v10/v11/v12/v13/v26 detection | anchor-free / DFL-style detection, no objectness field | `yolov8`, `yolov9`, `yolov10`, `yolov11`, `yolov12`, `yolov13`, `yolov26` | `YOLOv8Postprocessor` (v8, v9, v11, v12, v13; `yolo11-n_640x640` uses `YOLOv11Postprocessor`), `YOLOv10Postprocessor` (v10), `YOLOv26Postprocessor` (v26) | `YOLOv8Postprocessor` family |
| YOLO segmentation | boxes + masks/prototypes | `yolov8seg`, `yolov26seg`, `yolov5seg` | task-specific segmentation postprocessor | task-specific segmentation postprocessor |
| YOLO pose | boxes + keypoints | `yolov5pose`, `yolov8pose`, `yolov26pose` | task-specific pose postprocessor | task-specific pose postprocessor |
| YOLO face detection | face boxes and optional landmarks | `yolov5face`, `yolov7face` | face-specific postprocessor | face-specific postprocessor |
| YOLO OBB | oriented boxes | `yolov26obb` | OBB postprocessor | OBB postprocessor |
| PPU YOLO variants | hardware-assisted postprocess output | `yolov5_ppu`, `yolov7_ppu`, `yolov8_ppu`, etc. | PPU postprocessor | PPU postprocessor |

!!! note "NOTE"  

    If the model is a new detection-only YOLO variant and its output is compatible with the YOLOv8+ anchor-free/DFL family, start with `--postprocessor yolov8`. If it has explicit objectness and anchor-grid decoding, start with `--postprocessor yolov5` or `--postprocessor yolov7`.

    The YOLOv12 and YOLOv13 examples do not use a postprocessor of their own (the C++ `YOLOv12Postprocessor` class exists but is unused): their outputs follow the YOLOv8 layout, so `--postprocessor yolov12` and `--postprocessor yolov13` select `YOLOv8Postprocessor`, as the registered `yolov12-*` and `yolov13-*` examples do.

---

## [Step 2] Generate a New YOLO Example

Use `scripts/add_model.sh` to create the C++ and Python example trees from the closest reference model.

```bash
# YOLOv8-style detection model
./scripts/add_model.sh yolo-custom_640x640 object_detection --postprocessor yolov8 --lang both

# YOLOv7-style anchor-based model
./scripts/add_model.sh yolov7-custom_1280x1280 object_detection --base-model yolov7_640x640 --input-size 1280x1280 --lang both

# YOLOX-style model
./scripts/add_model.sh custom-yolox_640x640 object_detection --postprocessor yolox --lang both
```

An example lives in the family/variant layout, which is where `add_model.sh` writes it. `<family>` groups related models. `<variant>` is the `.dxnn` stem, and that folder holds the factory, `config.json`, and the entry files.

```text
src/cpp_example/object_detection/<family>/<variant>/
├── factory/<variant>_factory.hpp
├── <variant>_sync.cpp
├── <variant>_async.cpp
└── config.json

src/python_example/object_detection/<family>/<variant>/
├── factory/<variant>_factory.py
├── config.json
├── <variant>_sync.py
├── <variant>_async.py
├── <variant>_sync_cpp_postprocess.py
└── <variant>_async_cpp_postprocess.py
```

After generation, review both language trees. Do not assume a generated factory is final; the selected template only gives a safe starting point.

`add_model.sh` adds the variant to `config/model_registry.json` and `config/test_models.conf`; pass `--no-register` for local experiments.

---

## [Step 3] Tune `config.json`

Most YOLO customization is done through the `config` object of each variant's `config.json`. The Python scripts load the `config.json` next to the script automatically. The C++ executables read it only when it is passed with `--config` (for example `./bin/<variant>_sync -m assets/models/<variant>.dxnn -i sample/img/sample_kitchen.jpg --config src/cpp_example/object_detection/<family>/<variant>/config.json`); without `--config` they use the defaults built into the factory. Put the keys inside `config`: C++ lets the members of `config` override top-level keys, while Python lets top-level keys override `config`. Keep C++ and Python configs aligned unless a language-specific path intentionally differs.

Common keys:

| Key | Meaning | Typical use |
|-----|---------|-------------|
| `obj_threshold` | objectness threshold | YOLOv5/v7/YOLOX-style models with objectness |
| `score_threshold` | class confidence threshold (Python reads it as `conf_threshold`; `conf_threshold` itself is ignored by C++) | all detection models |
| `nms_threshold` | IoU threshold for NMS | all detection models |
| `num_classes` | number of classes | custom datasets that are not COCO-80 |
| `class_names` | display labels (C++ only; the Python `DetectionVisualizer` labels come from `custom_labels` in the variant's `visualizer.kwargs`) | custom dataset labels |
| `strides` | output strides of the Python PPU decoders (`YOLOv5PPUPostprocessor`, `YOLOv7PPUPostprocessor`); the Python YOLOv5/YOLOX/YOLOv8 decoders derive or fix their strides | PPU variants with non-standard heads |

Example for a custom 3-class YOLOv5-style model:

```json
{
  "config": {
    "obj_threshold": 0.25,
    "score_threshold": 0.30,
    "nms_threshold": 0.45,
    "num_classes": 3,
    "class_names": ["person", "vehicle", "animal"]
  }
}
```

!!! note "NOTE"  

    Custom anchors are not read from `config.json`: C++ uses the postprocessor presets, and Python variants that need them set them in the variant's `custom_ops.py` (e.g., `yolov7-w6_1280x1280_nodecode`).

Example for a custom YOLOv8-style model:

```json
{
  "config": {
    "score_threshold": 0.30,
    "nms_threshold": 0.45,
    "num_classes": 3,
    "class_names": ["person", "vehicle", "animal"]
  }
}
```

!!! warning  

    `num_classes` must match the compiled model output. If the model was compiled with a different class count, changing only `config.json` will not fix the tensor shape.

---

## [Step 4] Review the Factory Code

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

Python YOLOv8-style pattern: the postprocessor class is set in the variant's `config.json`, and the family factory builds it through `_build("postprocessor", ...)`:

```json
"postprocessor": {
  "class": "YOLOv8Postprocessor",
  "args": [{"token": "input_width"}, {"token": "input_height"}, {"token": "config"}]
}
```

Check these points after generation:

- `get_model_name()` returns the new model name.
- `get_task_type()` remains `object_detection` for detection models.
- C++ and Python use the same postprocessor family.
- Thresholds and class metadata are loaded from `config.json` (C++: only with `--config`).
- Specialized tasks use the correct task-specific visualizer and runner.

---

## [Step 5] Understand Output Shape Compatibility

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

## [Step 6] Build and Run

Build the generated targets:

```bash
./build.sh --type debug --target <variant>_sync
./build.sh --type debug --target <variant>_async
```

Run C++ examples:

```bash
./bin/<variant>_sync  -m assets/models/<variant>.dxnn -i sample/img/sample_kitchen.jpg --no-display -l 1 --config src/cpp_example/object_detection/<family>/<variant>/config.json
./bin/<variant>_async -m assets/models/<variant>.dxnn -v assets/videos/dance-group.mov --no-display -l 1 --config src/cpp_example/object_detection/<family>/<variant>/config.json
```

Run Python examples:

```bash
python src/python_example/object_detection/<family>/<variant>/<variant>_sync.py \
  --model assets/models/<variant>.dxnn \
  --image sample/img/sample_kitchen.jpg \
  --no-display \
  --loop 1

python src/python_example/object_detection/<family>/<variant>/<variant>_async.py \
  --model assets/models/<variant>.dxnn \
  --video assets/videos/dance-group.mov \
  --no-display \
  --loop 1
```

Use saved output for visual checks:

```bash
./bin/<variant>_sync -m assets/models/<variant>.dxnn -i sample/img/sample_kitchen.jpg --save --config src/cpp_example/object_detection/<family>/<variant>/config.json
```

---

## [Step 7] Verify Before Registry Integration

Before registering a new YOLO model as a maintained example, verify it in both languages.

Recommended sequence:

```bash
# Registry-driven validation (interactive; all registered models)
./scripts/dx_tool.sh validate

# Targeted build
./build.sh --type debug --target <variant>_sync
./build.sh --type debug --target <variant>_async

# CLI smoke tests
./scripts/dx_tool.sh run --lang cpp --filter <variant>
./scripts/dx_tool.sh run --lang py --filter <variant>

# Broader checks when assets are available
./run_tc.sh --cpp --cli
./run_tc.sh --python
```

For numerical validation, add or update rules only after the visual and CLI checks are stable. Avoid checking in model metadata that points to missing `.dxnn` assets.

---

## [Step 8] Update Registry and Test Metadata

When the model is intended to ship as a first-class example, review the registry and `test_models.conf` rows that `add_model.sh` added, and update the rest together:

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

Do not add registry entries for local experiments or project-specific artifacts unless the release explicitly includes them.

---

## Troubleshooting

### No detections appear

- Lower `score_threshold` temporarily.
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

- Ensure both configs contain the same thresholds, class count, and labels, and that the C++ run passes its `config.json` with `--config`.
- Ensure both factories use the same postprocessor family.
- Run one image with display disabled and compare saved output or serialized verification output.
- Check whether one path uses a C++ binding variant while the other uses pure Python postprocess.

### Build target is missing

- Confirm the family directory is `<family>/` and the variant directory is the `.dxnn` stem.
- Run `./build.sh --target list` to confirm the target exists.
- Check that CMake discovered the generated example directory.

---

## Release Checklist

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

---
