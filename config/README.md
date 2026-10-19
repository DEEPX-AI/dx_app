# config/ — DX-APP Configuration Files

Two configuration files that define **which models exist** and **how to run them**.

---

## `model_registry.json` — Single Source of Truth

A JSON array containing metadata for every supported model. This file is the authoritative reference for model code generation and validation.

### Format

An actual row (`yolov8-n_640x640`):

```json
{
  "model_name": "yolov8n",
  "dxnn_file": "yolov8-n_640x640.dxnn",
  "original_name": "YoloV8N",
  "csv_task": "OD",
  "add_model_task": "object_detection",
  "postprocessor": "yolov8",
  "input_width": 640,
  "input_height": 640,
  "config": {
    "score_threshold": 0.25,
    "nms_threshold": 0.45
  },
  "source": "csv",
  "supported": true,
  "variant": "yolov8-n_640x640",
  "family": "yolov8",
  "task": "object_detection",
  "task_legacy": "object_detection",
  "image_only": false,
  "zoo_canonical": true,
  "alias_of": null,
  "published": true
}
```

The model key is the **variant**: its example lives in
`src/{cpp,python}_example/<task>/<family>/<variant>/` (factory, `config.json`
and entry files), and its model file is `assets/models/<variant>.dxnn`. A
row's `model_name`, where it differs from the variant, is the variant's old
name; it still resolves to the variant (the graph CLI prints a note), but it
is not a second model.

### Where It Is Used

| Script | Purpose |
|---|---|
| `scripts/check_model_registry.py` | Guard: every row's `<task>/<family>/<variant>/` holds a C++ and a Python factory and every factory is registered; one row per variant; no name resolves two ways (`scripts/ci_checks.sh`: `guard-model-registry`) |
| `scripts/gen_model_registry.py` | Generates the graph engine's model registry and `docs/graph_models.md` from the rows and the factory tree (run by the CMake configure step) |
| `scripts/download_models.py` / `setup.sh --models` | Resolves a model name, old name or `.dxnn` stem to the model-zoo download |
| `scripts/regen_test_models_conf.py` | Regenerates `test_models.conf` |
| C++ and Python runners | An example run without `-m` takes its variant's `dxnn_file` from here |
| `tests/` (pytest) | The authoritative list of models to test |

### Field Reference

| Field | Description |
|---|---|
| `variant` | The model's identity: the `.dxnn` stem and the name of its example directory `<task>/<family>/<variant>/` |
| `family` | The family directory that groups related variants (e.g. `yolov8`) |
| `task` | The AI task directory (e.g. `object_detection`, `image_classification`) |
| `model_name` | The row's name. Equal to `variant`, or the variant's old name, which still resolves to it |
| `alias_of` | `null`, or the `model_name` of the row this row aliases: an alias row names the same variant and `.dxnn` as that row and is not a second model (`deit_base384_distilled` is the one alias) |
| `published` | `true` when the model zoo publishes the `.dxnn`. Every current row is `true`. `false` is reserved for a declared model that is not downloadable yet (manifest `pending: true`) |
| `dxnn_file` | Filename inside `assets/models/` (`<variant>.dxnn`) — case must match exactly |
| `original_name` | The model zoo's display name (e.g., `YoloV8N`) |
| `input_width` / `input_height` | Model input size in pixels |
| `config` | Per-model runtime parameters (thresholds, `num_classes`, etc.) |
| `postprocessor` | `--postprocessor` value passed to `add_model.sh` |
| `add_model_task` | Task value passed to `add_model.sh`; see [Categories](#categories-add_model_task) |
| `task_legacy` | The `add_model_task` value from before the task/family/variant layout, kept verbatim |
| `csv_task` | Short task **label** for CSV export / model-zoo bookkeeping (e.g., `OD`, `IC`, `SEG`). **Metadata only** — not read by any example code or runtime script |
| `image_only` | `true` for a model whose examples take only `--image` (no video, camera or RTSP input) |
| `zoo_canonical` | `true` when the variant is in the DX Model Zoo release tree (`tests/data/modelzoo_cv_tree.json`) |
| `source` | Origin of the row: `csv`, `inferred`, `manifest`, `auto_classify`, or `manual` |
| `supported` | Set to `false` to skip this model during validation (add `failure_reason` to note why) |

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

A tab-separated file, generated from `model_registry.json` by `python3 scripts/regen_test_models_conf.py`, that maps each variant to its family, task and compiled `.dxnn` path. Used by the example runner and benchmark scripts to determine which model file to load and which default inputs to use.

### Format

```
# family<TAB>task<TAB>model_file<TAB>variant
yolov8	object_detection	assets/models/yolov8-n_640x640.dxnn	yolov8-n_640x640
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
  regen_test_models_conf.py      (configure execution)
       │
       ▼
  run_examples.sh / bench_models.sh  (run / benchmark)
```

`add_model.sh --auto-add` generates the example of every registry row that has
none, under `src/{cpp,python}_example/<task>/<family>/<variant>/`; a single
`add_model.sh <variant> <task>` also adds the row and the `test_models.conf`
line for a variant that is not registered yet.

---

## Important Notes

- `dxnn_file` must match the actual filename inside `assets/models/` **exactly, including letter case**.
- The `variant` column of `test_models.conf` is the registry's `variant`; regenerate the file with `python3 scripts/regen_test_models_conf.py` after a registry change instead of editing it.
- When adding a new model, register it in the registry first (`variant`, `family`, `task`, `dxnn_file`), put its example under `src/{cpp,python}_example/<task>/<family>/<variant>/`, then regenerate `test_models.conf`; `python3 scripts/check_model_registry.py` checks that the rows and the factory tree agree.
- `input_width`/`input_height` should match the **compiled `.dxnn` input tensor** (as reported by `dxrt-cli` model parsing / the model-zoo "Input Resolution"), not the nominal paper resolution — e.g. `RegNetY16GF` compiles at `384×384`, `ulfgfd-RFB-640` at `640×480`. For non-square models the order is **W×H** (`input_width` first).
- `csv_task` / `add_model_task` / (`category` in the Model Zoo manifest) should agree. If they disagree, trust the **example directory** + `test_models.conf` (what actually runs); the manifest category may be less precise.
