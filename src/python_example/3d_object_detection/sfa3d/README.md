# SFA3D 608x608 — LiDAR 3D object detection example

This example converts a KITTI LiDAR point cloud to a bird's-eye view (BEV) and
detects 3D objects with SFA3D. It provides 4 Python variants and 2 C++ variants.

| Item | Value |
|------|-------|
| Model | `sfa3d_608x608` (`sfa3d_608x608.dxnn`) |
| Task type | `3d_object_detection` |
| Input | LiDAR point cloud (`.bin`) — KITTI velodyne format |
| Input resolution | 608 × 608 (BEV) |
| Classes | `Pedestrian`, `Car`, `Cyclist` |

---

## 1. Prerequisites

### 1-1. Build

```bash
./install.sh && ./build.sh      # C++ examples + dx_postprocess pybind11 (Linux)
# Windows: build.bat --minimal
```

- The C++ examples (`sfa3d_608x608_sync` / `_async`) are produced by that build.
- The Python `*_cpp_postprocess.py` variants need the `dx_postprocess` pybind11 module (included in the build).
  > **Important:** after editing the SFA3D postprocess (`src/postprocess/sfa3d/`), **rebuild** `dx_postprocess.so`.
  > A stale module produces wrong `z3d` / `dim` values and breaks the Cam+Box view.

### 1-2. Download the model

```bash
./setup.sh --models sfa3d_608x608     # → assets/models/sfa3d_608x608.dxnn
```

### 1-3. Check the NPU

```bash
dxrt-cli -s
```

---

## 2. Layout

```
src/python_example/3d_object_detection/sfa3d/
├── README.md
├── calib_policy.py                                 # companion calib-file policy
└── sfa3d_608x608/
    ├── config.json                                 # score / nms thresholds
    ├── factory/sfa3d_608x608_factory.py            # IDetectionFactory
    ├── sfa3d_608x608_sync.py                       # 1 Python sync
    ├── sfa3d_608x608_async.py                      # 2 Python async
    ├── sfa3d_608x608_sync_cpp_postprocess.py       # 3 Python sync + C++ postprocess
    └── sfa3d_608x608_async_cpp_postprocess.py      # 4 Python async + C++ postprocess

src/cpp_example/3d_object_detection/sfa3d/sfa3d_608x608/
├── config.json
├── factory/sfa3d_608x608_factory.hpp               # I3DDetectionFactory
├── sfa3d_608x608_sync.cpp                          # 5 C++ sync
└── sfa3d_608x608_async.cpp                         # 6 C++ async
```

---

## 3. How to run

### 3-1. Python (run from the variant directory)

```bash
cd src/python_example/3d_object_detection/sfa3d/sfa3d_608x608

# 1 sync — one .bin
python sfa3d_608x608_sync.py \
  -m ../../../../../assets/models/sfa3d_608x608.dxnn \
  -i ../../../../../sample/kitti/velodyne/000049.bin

# 2 async — velodyne directory batch
python sfa3d_608x608_async.py \
  -m ../../../../../assets/models/sfa3d_608x608.dxnn \
  -i ../../../../../sample/kitti/velodyne

# 3 / 4 C++ postprocess variants (dx_postprocess required; falls back to the Python postprocessor if it is missing)
python sfa3d_608x608_sync_cpp_postprocess.py  -m <model.dxnn> -i <bin or velodyne directory>
python sfa3d_608x608_async_cpp_postprocess.py -m <model.dxnn> -i <velodyne directory>
```

If the input is omitted, the default sample `sample/kitti/velodyne/000049.bin` is used.

### 3-2. C++ (built binaries)

```bash
# sync — one .bin
./sfa3d_608x608_sync \
  -m assets/models/sfa3d_608x608.dxnn \
  -i sample/kitti/velodyne/000049.bin

# async — velodyne directory batch, save, headless
./sfa3d_608x608_async \
  -m assets/models/sfa3d_608x608.dxnn \
  -i sample/kitti/velodyne \
  --calib-dir sample/kitti/calib \
  --image2-dir sample/kitti/image_2 \
  --save --no-display
```

---

## 4. CLI options

| Option | Description |
|--------|-------------|
| `-m`, `--model` | path to the `.dxnn` model (default: `assets/models/sfa3d_608x608.dxnn`) |
| `-i`, `--image` | one LiDAR `.bin` file **or** a velodyne directory |
| `--calib-dir` | directory of `{frame_id}.txt` calib files (paired with the `-i` stem) |
| `--image2-dir` | directory of `{frame_id}.png/.jpg` camera images (paired with the `-i` stem) |
| `--save`, `-s` | save the result image |
| `--no-display` | disable window output (headless) |
| `--config` | path to `config.json` (auto-detected by default) |
| `--loop`, `-l` | inference repeat count |
| `--show-log` | per-frame verbose log (quiet by default) |

> With the default KITTI layout (`velodyne/` + `calib/` + `image_2/`),
> `--calib-dir` and `--image2-dir` can be omitted.
> Both Python and C++ accept `-h` / `--help`. The C++ binary spells the long
> options `--model_path` / `--image_path`; the short forms are the same.

---

## 5. Input / samples

- **Supported input:** `--image` (one `.bin` file or a velodyne directory)
- **Unsupported input:** `--video`, `--camera`, `--rtsp` (LiDAR-only pipeline)

| Frame | Description |
|-------|-------------|
| `000049` | Car — default demo |
| `000535` | Car / Pedestrian |

Sample data: `sample/kitti/{velodyne,calib,image_2,label_2}/`

---

## 6. `config.json`

```json
{
    "score_threshold": 0.3,
    "nms_threshold": 0.2,
    "max_detections": 100,
    "require_calib": false
}
```

| Key | Description |
|-----|-------------|
| `score_threshold` | detection confidence threshold |
| `nms_threshold` | NMS IoU threshold |
| `max_detections` | maximum number of detections |
| `require_calib` | `true` raises an error when the calib file is missing (default `false`) |

---

## 7. Visualization (BEV)

- The **model-input BEV** is not modified (preprocessor grid: +y maps to the right column).
- For **display only**, the BEV raster is flipped horizontally first, then boxes, the legend, and labels are drawn in display coordinates `col = (y_max - y)`.
- Panel titles (for example `"BEV"`) are drawn after the flip, so the text is not mirrored.
- When both calib (`--calib-dir`) and a camera image (`--image2-dir`) are present, a Cam+Box (perspective projection) view is rendered as well.

---

## 8. `--show-log`

Same behavior as the other dx_app examples.

| Output | without `--show-log` | with `--show-log` |
|--------|----------------------|-------------------|
| Model loaded / input size | yes | yes |
| Starting inference | yes | yes |
| PERFORMANCE SUMMARY | yes | yes |
| Config loaded | no | yes |
| Input path / resolution | no | yes |
| `[Result] Detected N...` / detection coordinates | no | yes |
| Per-frame Read / Pre / Infer milliseconds | no | yes |

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `dx_postprocess.SFA3DPostProcess not available` | The pybind module is not installed, so the example falls back to the Python postprocessor. Rebuild `dx_postprocess` with `./build.sh` to use the C++ postprocess. |
| Cam+Box view is broken | A stale `dx_postprocess.so` is loaded. Rebuild the module after editing the postprocess. |
| Model not found | Download it with `./setup.sh --models sfa3d_608x608` and check the `-m` path. |
| `--video` / `--camera` do nothing | SFA3D accepts only LiDAR `.bin` input. That limit is intentional. |
