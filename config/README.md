# config/ — DX-APP Configuration Files

Two configuration files that define **which models exist** and **how to run them**.

---

## `model_registry.json` — Single Source of Truth

A JSON array containing metadata for every supported model. This file is the authoritative reference for model code generation and validation.

### Format

```json
{
  "model_name": "yolov8n",
  "add_model_task": "object_detection",
  "postprocessor": "yolov8",
  "dxnn_file": "YoloV8N.dxnn",
  "supported": true,
  "config_overrides": {
    "score_threshold": 0.25,
    "nms_threshold": 0.45,
    "num_classes": 80
  }
}
```

### Where It Is Used

| Script | Purpose |
|---|---|
| `scripts/validate_models.sh` | Iterates all entries and runs `add_model.sh --verify` for each |
| `scripts/add_model.sh` | Reads `config_overrides` to generate per-model `config.json` |
| `scripts/add_model.sh --auto-add` | Batch-generates source packages for all unregistered models |
| `tests/` (pytest) | Used as the authoritative list of models to test |

### Field Reference

| Field | Required | Description |
|---|---|---|
| `model_name` | ✅ | Lowercase model identifier — must match the example directory name |
| `add_model_task` | ✅ | Task type passed to `add_model.sh` (e.g., `detection`, `classification`) |
| `postprocessor` | ✅ | `--postprocessor` value passed to `add_model.sh` |
| `dxnn_file` | ✅ | Filename inside `assets/models/` — case must match exactly |
| `supported` | | Set to `false` to skip this model during validation |
| `config_overrides` | | Per-model runtime parameters (thresholds, `num_classes`, etc.) |

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
| `scripts/run_examples.sh` | Maps each model to its `.dxnn` path and category-specific default input |
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
