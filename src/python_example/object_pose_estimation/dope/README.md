# DOPE Hope-Ketchup — 6DoF Object Pose Estimation example (demo)

This example detects the 8 cuboid vertices plus the centroid from belief maps,
estimates a 6DoF pose with PnP, and overlays the 3D cuboid on the image. It
provides 4 Python variants and 2 C++ variants.

| Item | Value |
|------|-------|
| Model | `dope_hope_ketchup_1` (`dope-hope-ketchup_480x640.dxnn`) |
| Task type | `object_pose_estimation` |
| Input | UINT8 image `[1, 480, 640, 3]` |
| Input resolution | 640 × 480 |
| Output | `[1, 25, 60, 80]` — ch 0–8 belief maps (8 vertices + centroid), ch 9–24 affinity fields |
| Object | Hope-Ketchup, single object |

---

## Demo scope and limits

This example is a **visualization demo**. It follows a simplified `dope_decode`
path rather than the full postprocess of
[NVlabs/Deep_Object_Pose](https://github.com/NVlabs/Deep_Object_Pose).

| Item | This example (demo) | NVlabs original |
|------|---------------------|-----------------|
| Peak detection | **one argmax** per belief-map channel | **multiple peaks** via local maxima after Gaussian blur (σ=3) |
| Affinity fields (ch 9–24) | **unused** | vertex-to-centroid vector voting (association) |
| Multiple objects | **not supported (single instance only)** | multi-object association |
| Detection gating | `conf_threshold` (centroid confidence) | `thresh_map` / `thresh_points` / `thresh_angle` |
| Camera intrinsics | **estimated** (focal = image width) | real calibration (`camera_info.yaml`) |
| Pose accuracy | the overlay looks plausible, but **metric accuracy is not guaranteed** | metric accuracy with calibration |

> One object in the frame is enough for a visual demo. Multiple objects or a
> metric pose need affinity-based association and a real camera calibration.

---

## 1. Prerequisites

```bash
./install.sh && ./build.sh                 # C++ examples + dx_postprocess pybind11
./setup.sh --models dope_hope_ketchup_1     # download the model
dxrt-cli -s                                 # check the NPU
```

> The `*_cpp_postprocess.py` variants use the `dx_postprocess` pybind11 module.
> After editing `src/postprocess/dope/`, **rebuild** the module so the change takes effect.

---

## 2. Layout

```
src/python_example/object_pose_estimation/dope/
├── README.md
└── dope-hope-ketchup_480x640/
    ├── config.json                                              # tuning parameters (section 5)
    ├── factory/dope-hope-ketchup_480x640_factory.py             # IPoseFactory
    ├── dope-hope-ketchup_480x640_sync.py                        # 1 Python sync
    ├── dope-hope-ketchup_480x640_async.py                       # 2 Python async
    ├── dope-hope-ketchup_480x640_sync_cpp_postprocess.py        # 3 Python sync + C++ postprocess
    └── dope-hope-ketchup_480x640_async_cpp_postprocess.py       # 4 Python async + C++ postprocess

# Shared components
src/python_example/common/processors/dope_postprocessor.py   # peak detection + PnP
src/python_example/common/processors/_dope_geometry.py       # 3D cuboid / camera / PnP helpers
src/python_example/common/visualizers/dope_visualizer.py     # cuboid overlay

# C++
src/postprocess/dope/                                        # C++ peak postprocess
src/cpp_example/object_pose_estimation/dope/dope-hope-ketchup_480x640/
```

---

## 3. How to run

### 3-1. Python (run from the variant directory)

```bash
cd src/python_example/object_pose_estimation/dope/dope-hope-ketchup_480x640

# 1 sync — single image
python dope-hope-ketchup_480x640_sync.py \
  -m ../../../../../assets/models/dope-hope-ketchup_480x640.dxnn \
  -i <image.jpg>

# 2 async
python dope-hope-ketchup_480x640_async.py  -m <model.dxnn> -i <image or directory>

# 3 / 4 C++ postprocess variants (dx_postprocess module required)
python dope-hope-ketchup_480x640_sync_cpp_postprocess.py   -m <model.dxnn> -i <image>
python dope-hope-ketchup_480x640_async_cpp_postprocess.py  -m <model.dxnn> -i <directory>
```

If the input is omitted, the task default sample image is used. If `--config` is
omitted, `config.json` in this folder is applied automatically. Use `-h` /
`--help` for the full option list.

### 3-2. C++ (built binaries)

> **Build first.** `object_pose_estimation` is in `CATEGORIES` in
> `src/cpp_example/CMakeLists.txt`, so `./build.sh` builds the DOPE C++
> example (`solvePnP` / `projectPoints` need `opencv_calib3d`). The binaries
> are `dope-hope-ketchup_480x640_sync` and `dope-hope-ketchup_480x640_async`.

```bash
# 5 sync — single image
./dope-hope-ketchup_480x640_sync \
  -m assets/models/dope-hope-ketchup_480x640.dxnn \
  -i <image.jpg>

# 6 async — image directory
./dope-hope-ketchup_480x640_async \
  -m assets/models/dope-hope-ketchup_480x640.dxnn \
  -i <image_dir>

# save, headless (write results without opening a window)
./dope-hope-ketchup_480x640_async -m <model.dxnn> -i <image_dir> --save --no-display
```

| Option | Description |
|--------|-------------|
| `-m`, `--model_path` | path to the `.dxnn` model (default: `assets/models/dope-hope-ketchup_480x640.dxnn`) |
| `-i`, `--image_path` | image file **or** directory |
| `-v`, `--video_path` | accepted, but refused for this task (image input only) |
| `-c`, `--camera_index` | accepted, but refused for this task (image input only) |
| `-r`, `--rtsp_url` | accepted, but refused for this task (image input only) |
| `-s`, `--save` | save results to disk |
| `--save-dir` | base directory for saved results (default `artifacts/cpp_example`) |
| `--no-display` | disable window output (headless) |
| `--config` | path to `config.json` |
| `-l`, `--loop` | inference repeat count |
| `--show-log` | verbose log (quiet by default) |
| `-h`, `--help` | option help |

> Specify **exactly one** input source (`-i` / `-v` / `-c` / `-r`).
> If every input is omitted, the task default sample image is used.

---

## 4. Output (`DopeResult`)

One detection returned by the Python postprocessor (`DOPEPostprocessor`):

| Field | Contents |
|-------|----------|
| `keypoints` | `(9, 2)` normalized `[0, 1]` coordinates (8 vertices + centroid) |
| `centroid` | `(2,)` normalized centroid |
| `confidence` | centroid belief peak |
| `all_conf` | `(9,)` per-channel peak values |
| `pose` | `{'R': (3, 3), 't': (3,), 'rvec': (3, 1), 'tvec': (3, 1)}`, or `None` |
| `image_width` / `image_height` | original image size (for pixel conversion) |

> `pose` is filled when `solve_pnp=true` and at least 4 vertices are valid.
> The visualizer reuses that `pose` to project the cuboid, so the drawn box
> and `pose` always match.

---

## 5. `config.json` — tuning parameters

```json
{
    "conf_threshold": 0.0,
    "subpixel_refine": true,
    "solve_pnp": true,
    "object_size_cm": [14.860799789428711, 4.3368000984191895, 6.4513998031616211],
    "focal_length": null
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `conf_threshold` | `0.0` | Drop the detection when the centroid belief peak is below this value. **`0.0` means "always detect" (demo default).** Use `0.1`–`0.3` to reduce false boxes when no object is present. Without affinity / `thresh_map`, this is the only gate. |
| `subpixel_refine` | `true` | Sub-pixel correction from a 3×3 belief-weighted centroid around the argmax. `false` uses the integer peak. |
| `solve_pnp` | `true` | `false` skips PnP and sets `pose=None` (keypoints only). |
| `object_size_cm` | ketchup `[14.86, 4.34, 6.45]` | Real cuboid size `[width, height, depth]` in centimeters. **Replace this for any other object**, or the pose will be wrong. |
| `focal_length` | `null` | Camera focal length in pixels. `null` uses the demo estimate (image width). **Set a real calibration value when a metric pose is required** (section 6). |

> The factory injects the same `object_size_cm` and `focal_length` into both the
> postprocessor and the visualizer. Changing only one of them makes the overlay
> and `pose` disagree.

---

## 6. Camera intrinsics — real calibration

The default camera matrix is an **estimate**:

```
K = [[ f, 0, W/2 ],
     [ 0, f, H/2 ],   f = focal_length (default: image width W)
     [ 0, 0,  1  ]]   principal point = image center, distortion = 0
```

- The cuboid overlay can still look plausible, but the **translation (`t`) and
  rotation (`R`) in `pose` are not metrically accurate**.
- For a real 6DoF pose:
  1. Calibrate the camera and obtain focal length (px), principal point, and distortion.
  2. Set `focal_length` in `config.json` to the focal length in pixels.
  3. If the principal point is not the image center, or distortion is large,
     replace `K` and `dist` in `_dope_geometry.build_camera_matrix()` /
     `solve_pose()` with the calibration values. This demo assumes a centered
     principal point and zero distortion.

---

## 7. Visualization

- 8 vertices (colored points) plus the centroid (white point)
- On a successful PnP, 12 cuboid edges (green) and an X on the top face (yellow)
- If fewer than 4 vertices are valid, the detected peaks are connected directly as a fallback

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| A box is drawn when no object is present | The demo has no affinity / `thresh_map` gating. Raise `conf_threshold` in `config.json` (for example `0.2`). |
| The box breaks when two or more objects are present | Single-instance only (one argmax per channel). Multi-object support needs affinity association and is outside this demo. |
| Cuboid size or shape does not match | `object_size_cm` does not match the object. Replace it with the real dimensions. |
| Pose distance looks unrealistic | `focal_length` is the estimate (image width). Set the calibrated focal length (section 6). |
| `pose` is always `None` | `solve_pnp` is `false`, or fewer than 4 vertices are valid. Check the config. |
| C++ postprocess results differ | `src/postprocess/dope/` was edited but `dx_postprocess` was not rebuilt. Run `./build.sh` again. |
