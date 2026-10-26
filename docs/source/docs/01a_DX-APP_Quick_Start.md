# DX-APP Quick Start

This chapter takes a fresh DX-APP installation to its first inference on a DEEPX NPU in five steps. Each step links to the chapter that explains it in detail.  

!!! note "Before you start"  

    The NPU device driver and **DX-RT 3.5.0 or later** must already be installed. The supported operating systems, the hardware requirements and the driver and DX-RT installation are described in [DX-APP Installation and Build](02_DX-APP_Installation_and_Build.md).  

---

## Step 1. Verify the NPU and DX-RT

```bash
# NPU device and driver status
dxcli -s

# DX-RT version: the first line must be DXRT v3.5.0 or later
dxcli --version
```

If `dxcli -s` fails, or DX-RT is older than 3.5.0, install or upgrade the NPU driver and DX-RT before you continue (*Installation on Linux* in [DX-APP Installation and Build](02_DX-APP_Installation_and_Build.md)). The models that `setup.sh` downloads need DX-RT 3.5.0, and an older runtime fails only when a model is loaded.  

---

## Step 2. Install the Dependencies

```bash
# Build tools, CMake and OpenCV
./install.sh --all
```

---

## Step 3. Download Models and Sample Videos

```bash
# Only the models that run_demo.sh and this chapter use
./setup.sh --demo-models

# Or choose categories and models from a menu
./setup.sh

# Or download every registered model, without prompts
./setup.sh --all
```

Models are saved in `assets/models/` and sample videos in `assets/videos/`. Every `setup.sh` option is listed in [DX-APP Installation and Build](02_DX-APP_Installation_and_Build.md).  

An example run without `-m` downloads its default model if it is missing, but sample videos are not downloaded on demand, so run this step before the video examples below.  

---

## Step 4. Build

```bash
# Default (minimal) build: the C++ examples run_demo.sh uses, and dx_postprocess
./build.sh

# Full build: every C++ example, the multi-model graph CLI and, when pybind11 is installed in the build's Python, dx_graph
./build.sh --all

# Only the targets you name
./build.sh --target yolov9-s_640x640_sync yolov9-s_640x640_async
```

Executables are written to `bin/`. `./build.sh --target list` lists the build targets (names that start with a digit, such as `3ddfa-v2_mobilenetv1_120x120_sync`, are not listed but build with `--target`), and `./build.sh --clean` rebuilds from scratch.  

---

## Step 5. Run an Example

The default build includes the YOLO11-N object detector, so this step uses it. Without `-m` (`--model` in Python), an example uses its default model, here `assets/models/yolo11-n_640x640.dxnn`; pass `-m <file>` to run another `.dxnn`.  

**C++**  

```bash
# Image, synchronous
./bin/yolo11-n_640x640_sync -i sample/img/sample_street.jpg

# Video, asynchronous (pipelined for throughput)
./bin/yolo11-n_640x640_async -v assets/videos/snowboard.mp4
```

**Python**  

```bash
# Image, synchronous, Python post-processing
python src/python_example/object_detection/yolo11/yolo11-n_640x640/yolo11-n_640x640_sync.py \
    --image sample/img/sample_street.jpg

# Video, asynchronous, C++ post-processing (dx_postprocess): the fastest Python variant
python src/python_example/object_detection/yolo11/yolo11-n_640x640/yolo11-n_640x640_async_cpp_postprocess.py \
    --video assets/videos/snowboard.mp4
```

Run the Python examples with the Python environment that has `dx_engine`, the DX-RT Python package. The default `./build.sh` installs `dx_postprocess` into the `python3` that is first on `PATH`: activate the `dx_engine` environment before you build, or pass `--venv_path <venv>` (or `--python_exec <python>`).  

**Result**  

A window shows the detections, and the console prints a performance summary (latency and FPS). On a machine without a display, add `--no-display`. To keep the result, add `--save`: it is written under `artifacts/`.  

---

## Next Steps

- **Interactive demo**: `./run_demo.sh` runs 26 demo tasks (detection, segmentation, pose, depth, super-resolution and more), one model each, on a sample image or video.  
- **Several models at once**: after `./build.sh --all`, `bash run_multi_model_demo.sh` runs one of three sample graph scenarios (hand cascade, logistics volume, DMS) from a menu; `--check` validates all 22 shipped graphs without the NPU. See [DX-APP Multi-Model Graph](06a_DX-APP_Multi_Model_Graph.md).  
- **Options and runtime features**: [DX-APP C++ Usage Guide](03_DX-APP_CPP_Example_Usage_Guide.md), [DX-APP Python Usage Guide](05_DX-APP_Python_Example_Usage_Guide.md) and [DX-APP Architecture and Usage Reference](09_DX-APP_Project_Overview.md).  
- **Your own model**: [DX Tool Guide](10_DX-APP_DX-Tool_Guide.md) and [DX-APP YOLO Customizing Guide](12_DX-APP_YOLO_Customizing_Guide.md).  
