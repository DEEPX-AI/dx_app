# config/ — DX-APP Configuration Files

Two configuration files that define **which models exist** and **how to run them**.

---

## `model_registry.json` — Single Source of Truth

A JSON array containing metadata for every supported model. This file is the authoritative reference for model code generation and validation.

### Format

```json
{
  "model_name": "yolov8n",
  "original_name": "YoloV8N",
  "csv_task": "OD",
  "add_model_task": "object_detection",
  "postprocessor": "yolov8",
  "dxnn_file": "YoloV8N.dxnn",
  "input_width": 640,
  "input_height": 640,
  "config": {
    "score_threshold": 0.25,
    "nms_threshold": 0.45,
    "num_classes": 80
  },
  "source": "csv",
  "supported": true
}
```

### Where It Is Used

| Script | Purpose |
|---|---|
| `scripts/validate_models.sh` | Iterates all entries and runs `add_model.sh --verify` for each |
| `scripts/add_model.sh` | Reads `config` to generate per-model `config.json` |
| `scripts/add_model.sh --auto-add` | Batch-generates source packages for all unregistered models |
| `tests/` (pytest) | Used as the authoritative list of models to test |

### Field Reference

| Field | Required | Description |
|---|---|---|
| `model_name` | ✅ | Lowercase model identifier — must match the example directory name |
| `original_name` | ✅ | Display name (e.g., `YoloV8N`) — used in UI/logs |
| `csv_task` | ✅ | Short task **label** for CSV export / model-zoo bookkeeping (e.g., `OD`, `IC`, `SEG`). **Metadata only** — not read by any example code or runtime script. See the [Task Code Reference](#task-code-reference-csv_task) below. |
| `add_model_task` | ✅ | Task type passed to `add_model.sh` — this **is** used by tooling to pick the example category, default inputs, and postprocessor family (e.g., `object_detection`, `classification`). See [Categories](#categories-add_model_task). |
| `postprocessor` | ✅ | `--postprocessor` value passed to `add_model.sh` |
| `dxnn_file` | ✅ | Filename inside `assets/models/` — case must match exactly |
| `input_width` | ✅ | Model input width in pixels |
| `input_height` | ✅ | Model input height in pixels |
| `config` | | Per-model runtime parameters (thresholds, `num_classes`, etc.) |
| `source` | | Origin of the model entry: `csv`, `inferred`, `manifest`, `auto_classify`, or `manual` |
| `supported` | | Set to `false` to skip this model during validation (add `failure_reason` to note why) |

### Categories (`add_model_task`)

This is the field that actually drives tooling (example directory, default input
media in `run_examples.sh`, and the postprocessor family in `add_model.sh`). Valid
values:

`object_detection`, `classification`, `pose_estimation`, `instance_segmentation`,
`semantic_segmentation`, `face_detection`, `depth_estimation`, `image_denoising`,
`image_enhancement`, `super_resolution`, `embedding`, `obb_detection`,
`hand_landmark`, `hand_detection`, `face_alignment`, `attribute_recognition`,
`reid`, `keypoint_detection`, `object_pose_estimation`,
`panoptic_driving_perception`, `3d_object_detection`, `ppu`

---

## `test_models.conf` — Execution Reference for Examples and Benchmarks

A tab-separated file that maps each model to its compiled `.dxnn` path and task category. Used by the example runner and benchmark scripts to determine which model file to load and which default inputs to use.

### Format

```
# model_name<TAB>category<TAB>model_file
yolov8n	object_detection	assets/models/YoloV8N.dxnn
```

### Where It Is Used

| Script | Purpose |
|---|---|
| `scripts/run_examples.sh` | Interactive or CLI execution of examples with per-model performance output |
| `scripts/bench_models.sh` | Selects models and `.dxnn` paths for benchmarking |

### Category → Default Input Mapping

| Category | Image | Video |
|---|---|---|
| `object_detection` | `sample/img/sample_street.jpg` | `assets/videos/dance-group.mov` |
| `face_detection` | `sample/img/sample_face.jpg` | `assets/videos/dance-solo.mov` |
| `pose_estimation` | `sample/img/sample_people.jpg` | `assets/videos/dance-solo.mov` |
| `classification` | `sample/img/sample_dog.jpg` | `assets/videos/dance-group.mov` |
| others | See comments at the top of the file | |

---

## Workflow

```
model_registry.json
       │
       ▼
  add_model.sh --auto-add       (generate source packages)
       │
       ▼
  validate_models.sh             (build + inference verification)
       │
       ▼
  register in test_models.conf   (configure execution)
       │
       ▼
  run_examples.sh / bench_models.sh  (run / benchmark)
```

---

## Important Notes

- `dxnn_file` must match the actual filename inside `assets/models/` **exactly, including letter case**.
- `model_name` must be **identical** between `model_registry.json` and `test_models.conf`.
- When adding a new model, register it in the registry first, then add the corresponding entry to `test_models.conf`.
- `input_width`/`input_height` should match the **compiled `.dxnn` input tensor** (as reported by `dxrt-cli` model parsing / the model-zoo "Input Resolution"), not the nominal paper resolution — e.g. `RegNetY16GF` compiles at `384×384`, `ulfgfd-RFB-640` at `640×480`. For non-square models the order is **W×H** (`input_width` first).
- `csv_task` / `add_model_task` / (`category` in the Model Zoo manifest) should agree. If they disagree, trust the **example directory** + `test_models.conf` (what actually runs); the manifest can be wrong (e.g. a hand *detector* mislabeled as *hand landmark*).
