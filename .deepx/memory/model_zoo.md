# Model Zoo — dx_app

> Compact reference for the model zoo. Do NOT enumerate models here —
> query `config/model_registry.json` as the single source of truth for current
> model counts and task types.

---

## Anti-Fabrication Warning

**NEVER guess or fabricate model names.** The registry uses lowercase `model_name`
values (e.g., `yolov8n`, `scrfd_10g`), NOT PascalCase class names. If you are unsure
whether a model exists, query the registry. Fabricating names causes silent failures
because the framework looks up `.dxnn` files by exact `model_name`.

**NEVER guess postprocessor values.** The `postprocessor` field in the registry is a
lowercase string key (e.g., `yolov8`, `scrfd`), NOT a Python or C++ class name.
The framework maps these keys to implementation classes internally.

---

## Source of Truth

```
config/model_registry.json
```

This is a **JSON array** of model entries (not an object). Each entry has:

| Field | Required | Description |
|---|---|---|
| `model_name` | yes | Unique lowercase identifier (e.g., `yolov8n`) |
| `dxnn_file` | yes | Compiled model filename (e.g., `yolov8n.dxnn`) |
| `add_model_task` | yes | Task type (see Task Overview below) |
| `postprocessor` | yes | Registry key for postprocessor (see mapping below) |
| `supported` | yes | `true` if model is available for use |
| `original_name` | no | Upstream model name |
| `csv_task` | no | CSV-compatible task label |
| `input_width` | no | Model input width in pixels |
| `input_height` | no | Model input height in pixels |
| `config` | no | Path to model-specific config JSON |
| `source` | no | Model provenance |

---

## Quick Queries

List all supported models:
```bash
jq '[.[] | select(.supported == true)] | length' config/model_registry.json
```

List models for a specific task:
```bash
jq '[.[] | select(.add_model_task == "object_detection" and .supported == true)] | .[].model_name' config/model_registry.json
```

Get a model's postprocessor and input size:
```bash
jq '.[] | select(.model_name == "yolov8n") | {postprocessor, input_width, input_height}' config/model_registry.json
```

List all distinct postprocessor keys:
```bash
jq '[.[].postprocessor] | unique' config/model_registry.json
```

List all distinct task types:
```bash
jq '[.[].add_model_task] | unique' config/model_registry.json
```

---

## Postprocessor Mapping

The `postprocessor` field in the registry is a **lowercase string key**. The framework
maps it to a Python class (for Python runner) and a C++ binding (for C++ runner).

| Registry Key | Python Class | C++ `dx_postprocess` | Notes |
|---|---|---|---|
| `yolov5` | `YOLOv5Postprocessor` | `YOLOv5PostProcess` | |
| `yolov8` | `YOLOv8Postprocessor` | `YOLOv8PostProcess` | |
| `yolov26` | `YOLOv8Postprocessor` | `YOLOv26PostProcess` | Python reuses YOLOv8 |
| `yolov10` | `YOLOv8Postprocessor` | `YOLOv10PostProcess` | Python reuses YOLOv8 |
| `yolox` | `YOLOXPostprocessor` | `YOLOXPostProcess` | |
| `damoyolo` | `DamoYoloPostprocessor` | `DamoYOLOPostProcess` | |
| `nanodet` | `NanoDetPostprocessor` | `NanoDetPostProcess` | |
| `ssd` | `SSDPostprocessor` | `SSDPostProcess` | |
| `efficientnet` | `ClassificationPostprocessor` | `ClassificationPostProcess` | All classifiers |
| `yolov8pose` | `YOLOv8PosePostprocessor` | `YOLOv8PosePostProcess` | |
| `yolov5seg` | `YOLOv5InstanceSegPostprocessor` | `YOLOv5SegPostProcess` | |
| `yolov8seg` | `YOLOv8InstanceSegPostprocessor` | `YOLOv8SegPostProcess` | |
| `yolact` | `YOLACTPostprocessor` | — | No C++ binding |
| `bisenetv1` | `SemanticSegmentationPostprocessor` | `SemanticSegPostProcess` | |
| `bisenetv2` | `SemanticSegmentationPostprocessor` | `SemanticSegPostProcess` | Shares with bisenetv1 |
| `deeplabv3` | `SemanticSegmentationPostprocessor` | `DeepLabv3PostProcess` | |
| `segformer` | `SegFormerPostprocessor` | `SemanticSegPostProcess` | |
| `scrfd` | `SCRFDPostprocessor` | `SCRFDPostProcess` | |
| `yolov5face` | `YOLOv5FacePostprocessor` | `YOLOv5FacePostProcess` | |
| `yolov7face` | `YOLOv7FacePostprocessor` | — | No C++ binding |
| `retinaface` | `RetinaFacePostprocessor` | `RetinaFacePostProcess` | |
| `fastdepth` | `DepthEstimationPostprocessor` | `DepthPostProcess` | |
| `dncnn` | `DnCNNPostprocessor` | `DnCNNPostProcess` | |
| `espcn` | `ESPCNPostprocessor` | `ESPCNPostProcess` | |
| `zero_dce` | `ZeroDCEPostprocessor` | `ZeroDCEPostProcess` | |
| `arcface` | `ArcFacePostprocessor` | `EmbeddingPostProcess` | |
| `obb` | `OBBPostprocessor` | `OBBPostProcess` | Score threshold only, no NMS |
| `yolov5_ppu` | `YOLOv5PPUPostprocessor` | `YOLOv5PPUPostProcess` | |
| `yolov7_ppu` | `YOLOv7PPUPostprocessor` | `YOLOv7PPUPostProcess` | |
| `hand_landmark` | `HandLandmarkPostprocessor` | — | No C++ binding |

**Key insight:** `yolov10` and `yolov26` share `YOLOv8Postprocessor` in Python but
have distinct C++ postprocessors. PPU models have dedicated postprocessors — never
use standard YOLO postprocessors with PPU models.

---

## Task Overview

The following task types are known (query `config/model_registry.json` for the current full list):

| Task | Postprocessor Keys Used | Example Models (query registry for full list) |
|---|---|---|
| `object_detection` | `yolov5`, `yolov8`, `yolov10`, `yolov26`, `yolox`, `damoyolo`, `nanodet`, `ssd` | yolov8n, yolov5s, ssd_mobilenetv2 |
| `classification` | `efficientnet` | efficientnet_b0, resnet50 |
| `pose_estimation` | `yolov8pose` | yolov8n_pose, yolov8s_pose |
| `instance_segmentation` | `yolov5seg`, `yolov8seg`, `yolact` | yolov8n_seg, yolov5s_seg |
| `semantic_segmentation` | `bisenetv1`, `bisenetv2`, `deeplabv3`, `segformer` | bisenetv1, deeplabv3plus |
| `face_detection` | `scrfd`, `yolov5face`, `yolov7face`, `retinaface` | scrfd_10g, retinaface |
| `depth_estimation` | `fastdepth` | fastdepth |
| `image_denoising` | `dncnn` | dncnn_15, dncnn_25, dncnn_50 |
| `super_resolution` | `espcn` | espcn_x2, espcn_x3, espcn_x4 |
| `image_enhancement` | `zero_dce` | zero_dce |
| `embedding` | `arcface` | arcface_mobilefacenet |
| `obb_detection` | `obb` | yolo26n_obb |
| `ppu` | `yolov5_ppu`, `yolov7_ppu` | yolov5s_ppu, yolov7_ppu |
| `hand_landmark` | `hand_landmark` | hand_landmark_lite |
| `attribute_recognition` | — | Query registry for models |
| `reid` | — | Query registry for models |

**Do NOT hardcode model names from this table.** These are examples only. Always
query `config/model_registry.json` for the current list.

---

## Common Mistakes

1. **Fabricating model names** — e.g., `yolov11n` or `efficientdet_d0` may not exist
   in the registry. Always verify with `jq`.

2. **Using PascalCase for postprocessor** — the registry field is a lowercase string
   like `yolov8`, not `YoloV8PostProcess`. The mapping to classes is internal.

3. **Wrong postprocessor family** — `segformer` models use `segformer` postprocessor,
   not `efficientnet`. Classification postprocessor is only for `classification` task.

4. **Ignoring `supported` field** — some models have `"supported": false`. Always
   filter by `supported == true` when listing available models.

5. **Mixing PPU and standard postprocessors** — PPU models (`yolov5_ppu`, `yolov7_ppu`)
   require their dedicated postprocessors. Standard YOLO postprocessors will fail.

6. **Assuming input sizes** — not all YOLO models use 640x640. Check `input_width` and
   `input_height` in the registry. Some models (e.g., nanodet, ssd) use smaller inputs.

7. **Hardcoding model lists** — the registry is the live source. This document
   intentionally does not list every model to avoid going stale.

## dx-modelzoo Alignment (Phase 0, 2026-09-16)

`config/model_registry.json` carries the authoritative (task, family, variant)
mapping, joined from a committed snapshot of the dx-modelzoo CV tree.

| Field | Meaning |
|---|---|
| `variant` | Identity for `--variant` and `assets/models/<variant>.dxnn`. Equals the dxnn stem except for the 3 documented exceptions. |
| `family` | dx-modelzoo family, snake_cased (`wide-resnet` → `wide_resnet`). 89 families. |
| `task` | dx-modelzoo task name. 23 values. |
| `task_legacy` | Pre-alignment `add_model_task`, preserved for migration. |
| `image_only` | Per-variant replacement for the task-keyed `_IMAGE_ONLY_TASKS`. 17 variants. |
| `zoo_canonical` | `false` for the entries absent from DX Model Zoo q-lite `2_4_0`. |
| `alias_of` | Set when an entry shares another entry's variant. Alias entries are **retained, never deleted**. |

- Snapshot: `tests/data/modelzoo_cv_tree.json` — refresh with
  `python scripts/harvest_modelzoo_tree.py`.
- Family assignment is **data, not an algorithm**: dx-modelzoo folds depth/width
  into the family (`resnet50` → `resnet`, `vgg16-bn` → `vgg`). Never derive it —
  a best-effort heuristic yields 211 families against the authoritative 89.
- **`variant` follows the `.dxnn`, never the legacy example-dir name.**
  `deit_base384_distilled` owns variant `deit-b_384x384_distilled` and is the
  only row for `deit-b_384x384_distilled.dxnn`. The legacy duplicate row
  `deit_base_distilled_2` was dropped, so that old name no longer resolves in
  `graph_models.md` — use the variant or `deit_base384_distilled`. The
  non-distilled file belongs only to `deitbase384`.
- **RETENTION POLICY — conflicting EXAMPLES are kept, never deleted.** The
  alignment renames and regroups; it does not remove examples. That is about the
  example tree, not the registry: a duplicate REGISTRY row may be dropped once its
  name is resolvable elsewhere. `deit-b_384x384_distilled.dxnn` was the only such
  conflict (2 rows) and the legacy `deit_base_distilled_2` row is now gone, so
  every row owns exactly one variant and conflict scope is 0. Differing `config` within a family (14 of 89)
  is NOT a conflict — each variant keeps its own `<family>/<variant>/config.json`.
- Guarded by `tests/python_example/unit/test_registry_modelzoo_conformance.py`
  and `test_registry_image_only.py`.
