# RELEASE_NOTES
## v3.1.0 / 2026-04-06

### 1. Changed
- Unified 5-layer architecture and design patterns across Python and C++ implementations
    - **App** Entry point -> `yolov5s_sync.cpp` : `yolov5s_sync.py`
    - **Runner** Pipeline orchestration (Sync/Async) -> `sync_detection_runner.hpp` : `sync_runner.py`
    - **Factory** Per-model component assembly -> `yolov5s_factory.hpp` : `yolov5s_factory.py`
    - **Component** Preprocessor / Postprocessor / Visualizer -> `processors/*.hpp` : `processors/*.py`
    - **Interface** Abstract contracts -> `i_factory.hpp`, `i_processor.hpp` : `i_factory.py`, `i_processor.py`
- Consolidated cross-language(Python : c++) common modules and 1:1 mapping structure
- Modernized `run_demo.sh` with a 3-stage interactive menu supporting variable AI tasks
- `--model`, `--image`, `--video` arguments are now optional — when omitted, task-appropriate default sample image/video is automatically selected
- `setup_sample_models.sh` migrated to Python-based downloader — supports `--list`, `--dry-run`, `--category`, `--models` and other granular download options
- `setup.sh` now integrates model download options — e.g. `setup.sh --models YoloV7 YoloV8S` to download specific models
- `run_demo.sh` fully redesigned — 18 demo models, 3-stage interactive menu (Task→Mode→Input), unified C++/Python support, `--task`/`--mode`/`--input` CLI arguments for non-interactive usage

### 2. Fixed

### 3. Added
- Supported new Depth Estimation task featuring FastDepth for monocular depth estimation
- Supported new Image Restoration task featuring DnCNN, Zero-DCE, and ESPCN models
- Migrated Full DX-Model Zoo encompassing 280 models across 17 taskcategories with 560 C++/Python examples (sync+async)
- Added yolov8, v9, v10, v11, v12 PPU models and C++/Python examples
- Implemented https://sdk.deepx.ai manifest-based DX-ModelZoo auto-download system (`scripts/download_models.py`)
- Auto-download for models and videos — automatically invokes `setup_sample_models.sh` when model file is missing, videos via `setup_sample_videos.sh`
- Interactive mode for `scripts/run_examples.sh` — 6-stage menu (Language→Category→Filter→ExecMode→InputType→Options) when run without arguments, with case-insensitive keyword filtering
- `dx_tool.sh run` unified with `run_examples.sh` interactive mode
- Real-time performance table output during example execution
- `--verbose` option for Python examples — controls per-frame detailed log output

---

## v3.0.4 / 2026-03-27

### 1. Changed

#### Test Infrastructure Restructuring
- **`tests/common/`**: extracted shared test constants (`constants.py`) and utilities (`utils.py`) from individual test files into a reusable module
- **Unified Visualization Tests**: consolidated per-task visualization test scripts into `tests/cpp_example/test_visualization.py` and `tests/python_example/test_visualization.py` — auto-discover all sync/async executables and scripts
- **Feature Test Suite**: added dedicated test modules for `--save` / `--save-dir` (`test_save_mode.py`), `--dump-tensors` (`test_dump_tensors.py`), `DXAPP_VERIFY` (`test_verify.py`), `--loop` (`test_multi_loop.py`), `SIGINT`/`SIGTERM` (`test_signal_handling.py`)
- Deleted 4 redundant root-level test scripts (`run_e2e_test.sh`, `run_visualization_test.sh`, `run_inference_test.sh`, `run_unit_test.sh`) — all functionality covered by `run_tc.sh`

#### Documentation Comprehensive Update
- **README.md**: added CLI Reference (full argument tables for C++/Python), Advanced Features section (signal handling, run directory, `DXAPP_VERIFY`, tensor dump, model config, version compatibility, headless mode), environment variables table, updated module counts and test tree
- **C++ Usage Guide** (`03`): added CLI Arguments table (12 flags), Advanced Features section, updated utility count (6→8)
- **Python Usage Guide** (`05`): added CLI Arguments table (13 flags), Advanced Features section, updated module counts (processors 34→35 description, visualizers 9→10 description)
- **Example Source Structure** (`11`): updated C++ utility count (6→8), Python utility count (6→7), added complete test infrastructure section with 9 test categories, `tests/common/` documentation
- **C++ Test README**: added visualization tests, feature tests (5 types), test infrastructure section, updated coverage summary
- **Python Test README**: expanded to all 14 task directories, added `test_visualization.py`, `tests/common/` reference

### 2. Added
- `tests/common/constants.py` — shared test constants (paths, timeouts, patterns)
- `tests/common/utils.py` — shared test utilities (discovery, execution, validation)
- `tests/cpp_example/test_visualization.py` — unified C++ visualization result tests
- `tests/cpp_example/test_save_mode.py` — `--save` / `--save-dir` flag tests
- `tests/cpp_example/test_dump_tensors.py` — `--dump-tensors` output tests
- `tests/cpp_example/test_verify.py` — `DXAPP_VERIFY` environment variable tests
- `tests/cpp_example/test_multi_loop.py` — `--loop` repeated execution tests
- `tests/cpp_example/test_signal_handling.py` — SIGINT/SIGTERM graceful shutdown tests
- `tests/python_example/test_visualization.py` — unified Python visualization result tests

### 3. Removed
- `run_e2e_test.sh` — replaced by `run_tc.sh --cpp --e2e`
- `run_visualization_test.sh` — replaced by `run_tc.sh --cpp --viz`
- `run_inference_test.sh` — replaced by `run_tc.sh --python`
- `run_unit_test.sh` — replaced by `run_tc.sh --unit`

---

## v3.0.3 / 2026-03-13

### 1. Changed
- Expanded model support across multiple AI task categories

### 2. Fixed

#### Post-Processing Bug Fixes (16 models)
- **Segmentation** (bisenetv1, bisenetv2, deeplabv3plusmobilenet): Fixed pre-argmaxed NPU output (uint16 `[1,H,W]`) being misinterpreted as logits — added heuristic detection for integer dtype / shape[0]==1
- **NanoDet** (nanodet_repvgg, nanodet_repvgga1): Fixed degenerate bounding boxes (y2==y1) caused by clipping to image boundary — added zero-area box filter
- **FastDepth** (fastdepth_1): Fixed `DepthResult` not handled by verify_serialize.py — added DepthResult serialization handler
- **YOLOv5Pose PPU** (yolov5pose_ppu): Fixed raw logit keypoint confidence (negative values) — applied sigmoid activation
- **YOLOX** (yoloxs, yoloxtiny, yolox_l_leaky, yolox_s_leaky, yolox_s_wide_leaky): Fixed zero bounding boxes from raw logit coordinates — implemented standalone `YOLOXPostprocessor` with grid decode (`cx=(cx_raw+grid_x)*stride`)
- **YOLOv7 Face** (yolov7_face, yolov7s_face, yolov7_w6_face, yolov7_w6_tta_face): Fixed confidence > 1.0 from misread column layout — added auto-dispatch by output column count (16-col vs 21-col) with sigmoid on raw class logit

### 3. Added

#### Shared Runtime Layer (`common/`)
- **C++ (`src/cpp_example/common/`)**: Base interfaces (`IFactory`, `IProcessor`, `IVisualizer`, `IInputSource`), 45 processors, 24 task-specific sync/async runner pairs, 12 visualizers, input source abstraction, config loader, utility
- **Python (`src/python_example/common/`)**: Base interfaces (`IFactory`, `IProcessor`, `IVisualizer`, `IInputSource`), 35 processors, generic `SyncRunner`/`AsyncRunner`, 10 visualizers, input source abstraction, `ModelConfig` loader, utility
- Both languages share the same 7-module architecture (`base/`, `config/`, `processors/`, `runner/`, `inputs/`, `visualizers/`, `utility/`) and factory-based delegation pattern

#### Model Registry System
- **`config/model_registry.json`**: centralized registry of 280 models with per-model metadata (task, postprocessor, input dimensions, thresholds)
- **`scripts/add_model.sh`**: registry-driven auto-generation of factory files, config.json, and entry-point scripts (4 variants per model)

#### Numerical Verification Framework
- **`scripts/verify_inference_output.py`**: 14 task-specific validators for bounding boxes, confidence ranges, class IDs, keypoints, segmentation masks, depth maps, embeddings
- **`scripts/inference_verify_rules.json`**: configurable thresholds per task type
- **`common/runner/verify_serialize.py`**: result-to-JSON serialization for automated comparison
- **`scripts/validate_models.sh --numerical`**: full-pipeline NPU verification for all supported models

#### New Model Families
- DAMOYOLO (5 variants), NanoDet (2), CenterNet, SSD MobileNet V1/V2, YOLOv3, YOLOv6
- FastDepth, MiDaS (depth estimation)
- CLIP, ArcFace (embedding)
- DnCNN, Zero-DCE (image denoising/enhancement)
- ESPCN (super resolution)
- Hand Landmark (2 variants)
- BiSeNet V1/V2, SegFormer (semantic segmentation)
- YOLOv7Face (4 variants), RetinaFace (face detection)
- YOLOv26 OBB, Seg, Pose, Cls variants

#### CI/CD Integration
- **`python-test.yml`**: automated unit/CLI/integration tests on PR (Python, no NPU required)
- **`npu-model-verify.yml`**: NPU numerical verification pipeline (self-hosted runner with NPU hardware)

## v3.0.2 / 2026-02-10

### 1. Changed
- Copy of dxrt and vkpkg DLLs into the dx-app/bin directory when building with MSVC.

### 2. Fixed
- Remove experimental filesystem includes and update float literals in example cpp files for build error on windows
- Refactor apply_argmax to reduce nesting and fix gcovr warnings

### 3. Added
- Added vcpkg installation script for windows build. 

## v3.0.1 / 2026-02-05

### 1. Changed

### 2. Fixed
- Hardcoded attribute size in YOLO post-processing to dynamically adjust based on model output shape

### 3. Added
- Add yolov26 cls, yolo26 pose, yolo26 seg, yolo26 obb examples

## DX-APP v3.0.0 / 2026-01-02

### Changed

#### Major Project Structure Refactoring
- **Complete overhaul from existing demo applications to example system**
  - To improve user understanding, separated the previously integrated example code by Task (classification, object detection, segmentation, face recognition, pose estimation) / Model (EfficientNet, YOLO, YOLO_PPU, SCRFD, ...) / Inference method (sync, async) / Post-processing (pure python, pybind)
    - Complete removal of legacy C++ demo code in `demos/` directory and provision of `run_demo.sh` and `run_demo.bat` based on separated examples
    - Transition to new `src/cpp_example/` and `src/python_example/` structure

#### Build System Improvements
- Improved CMake configuration and enhanced shared library support
- Updated C++17 and Visual Studio 2022(v143) configuration for Windows build
- Adjusted DXRT include and link directories for cross-compilation

#### Complete Reconstruction of C++ / Python Example System
- **Support for synchronous and asynchronous execution modes**
- **Support for various input sources**: image, video, camera, RTSP stream
- **Real-time processing mode**: Performance measurement without GUI using `--no-display` option
- **Enhanced performance profiling**: 
  - Latency measurement for each stage: preprocessing, inference, post-processing
  - E2E(End-to-End) FPS calculation and performance report generation
  - Automatic generation of timestamp-based performance report files

#### Model Support Expansion
- **YOLOv10, YOLOv11, YOLOv12** examples added
- **YOLOv8 Segmentation** (YOLOv8-seg) support
- **DeepLabv3** segmentation model support
- **PPU (Post-Processing Unit)** module integration:
  - YOLOv5, YOLOv7 PPU version support
  - SCRFD PPU version support
  - Both Python and C++ examples provided

#### Documentation Improvements
- Newly written example guides and installation guides
- Added detailed usage examples and parameter descriptions for each model

### Fixed

#### Code Quality Improvements
- Added try-catch error handling to all projects
- Improved `std::exception` handling and throw `std::invalid_argument` when layer requirements are not met
- Removed `using namespace std` usage and improved code clarity with explicit `std::` usage
- Improved parameter handling and frame processing logic
- Enhanced argument validation and error messages

#### Input Processing Improvements
- Set `cv2.CAP_PROP_BUFFERSIZE` (buffer size 1) for camera and RTSP speed improvement
- Fixed input_tensor passing to maintain memory reference until asynchronous inference completes

### Added

#### Post-processing Library (dx_postprocess)
- **Pybind11-based Python binding**:
  - Provides Python binding for C++ post-processing functions
  - Automatically installs to current Python execution environment

#### Multi-channel Processing Support
- C++ **YOLOv5s multi-channel processing**: Multi-channel support using frame provider
- Added multi-input source examples
- Enhanced multi-model image inference using preprocessing threads

#### Test Infrastructure Construction
- **Pytest-based integrated test system**:
  - Automated testing for all Python examples
  - Achieved code coverage of 93.65% or higher
  - E2E(End-to-End) test framework
  - Includes all model tests for classification, object detection, segmentation, and pose estimation
- Added `.coveragerc` file for code coverage configuration
- Support for display mode and E2E mode testing

#### New Examples and Features
- **Classification Models**:
  - EfficientNet example integration
  - ImageNet classification examples (synchronous/asynchronous)
  
- **Object Detection Models**:
  - YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10(python only), YOLOv11(python only), YOLOv12(python only)
  - YOLOX
  - SCRFD (face detection)
  - YOLOv5-Face
  - YOLOv5-Pose (pose estimation)

- **Segmentation Models**:
  - YOLOv8-seg (instance segmentation)
  - DeepLabv3 (semantic segmentation)

- **Pose Estimation**:
  - YOLOv5-Pose examples
  - Added skeleton drawing functionality

### Removed or Replaced

#### Legacy Demo Removal
- Complete removal of `demos/classification/`
- Complete removal of `demos/object_detection/`
- Complete removal of `demos/segmentation/`
- Complete removal of `demos/pose_estimation/`
- Complete removal of `demos/face_recognition/`
- Removal of `demos/denoiser/`
- Removal of `demos/dncnn_yolo/`
- Removal of `demos/object_det_and_seg/`
- Removal of `demos/noiseVideoMaker/`

#### Legacy Configuration File Removal
- Complete removal of JSON configuration files in `example/` directory
- Removal of `example/dx_postprocess/` JSON files
- Removal of Debian package related files (`debian/`)
- Removal of Docker build files (`docker/Dockerfile.app.build`)

#### Legacy Code Cleanup
- Removal of `demo_utils/` directory
- Removal of duplicate or unused code
- Removal of old YOLOv5 post-processing files
- Removal of RISCV64 architecture support

### Migration Guide

#### Notice for Existing Users
v3.0.0 is a major update that includes **Breaking Changes** compared to v2.x.

- **Demo Code**: The existing `demos/` directory has been completely removed. Please refer to the new examples in `src/cpp_example/` and `src/python_example/`.
- **JSON Configuration Files**: JSON files in the existing `example/` directory have been removed. Python examples are configured directly through command-line arguments.
- **YOLO Post-processing Type Names**: Some have been changed, but aliases are provided for backward compatibility.

#### Recommended Upgrade Path
1. Refer to Python example documentation
2. Check example code in `src/python_example/` or `src/cpp_example/` directory
3. Install Python dependencies through `requirements.txt`
4. Use build scripts (`build.sh` or `build.bat`)

### Known Issues
- When using the PPU model for face detection & pose estimation, `dx-compiler v2.1.0 and v2.2.0` does not currently support converting face and pose models to PPU format. This feature will be added in a future release. The PPU models used in the demo were converted using dx-compiler `v1.0.0(dx_com v1.60.1)`.

---

## DX-APP v2.1.0 / 2025-11-28

### Changed
- Enhance build script documentation and usage instructions
- Update cmake configuration in build.bat to use C++17 and v143 for enhance documentation windows build script(visual studio 2022)
- Model package updated from version 2.0.0 to 2.1.0 to support PPU models
- Improved demo script with additional PPU-Demo (1, 4, 6, 8, 11)
- Added CPU-specific PyTorch wheel source (https://download.pytorch.org/whl/cpu) in templates/python/requirements.txt.

### Fixed
- Fix Windows MSBuild compilation warnings by replacing implicit type casts with explicit static_cast
- Improve tensor allocation in imagenet classification example
- Update numBoxes calculation based on post-processing type in LayerReorder
- Rename YOLO post-processing types and add aliasing for backward compatibility
- Add VSCode configuration files for usability
- Fixed errors that occurred when using VAAPI with camera input
- Enhanced yolo application to display final FPS even when forcefully terminated during camera input usage
- Enhance user input handling for run_demo selection with re-prompt loops (invalid input re-asks instead of timing out)

### Added
- Windows Environment Support
DX-APP now fully supports the Windows operating system! In response to user requests, we've expanded compatibility beyond Linux to include Windows, enabling a broader range of development environments to take advantage of DX-APP.
    - **OS**: Windows 10 / 11
    - **Deepx M1 Driver Version**: v1.7.1 or higher
    - **Deepx M1 Runtime Lib Version**: v3.1.0 or higher
    - **Python**: Version 3.8 or higher (required for Python module support)
    - **Compiler**: Visual Studio Community 2022 (required for building C++ examples)
- Add automated build script (build.bat) for automatic build and Visual Studio solution generation
- Three new PPU data types : BBOX (for object detection) / POSE (for pose estimation keypoints) / FACE (for face detection landmarks)
- Enhanced post-processing functions to support PPU inference output format

### Known Issues
- DeepLabV3 Semantic Segmentation model accuracy may be slightly degraded in dx-compiler(dx_com) v2.1.0. This will be fixed in the next release. The DeepLabV3 model used in the demo was converted using dx-compiler v2.0.0.
- When using the PPU model for face detection & pose estimation, dx-compiler v2.1.0 does not currently support converting face and pose models to PPU format. This feature will be added in a future release. The PPU models used in the demo were converted using dx-compiler v1.0.0(dx_com v1.60.1).

---

## DX-APP v2.0.0 / 2025-08-14

### Changed
- Moved the YOLO post-processing guide from 07_Python_Examples.md to a new, dedicated document 08_YOLO_Post_Processing_Pybind11.md.
- Refactored yolo_pybind_example.py to use a RunAsync() + Wait() structure instead of callbacks. This ensures the output tensor order is correctly handled.
- Major code refactoring and restructuring of demo applications
- Consolidated common utilities into  directory
- Removed deprecated and legacy codes
- Update documentation and resources
- YoloPostProcess now filters and selects the correct tensor by output_name when USE_ORT=ON
- Command-line help messages in various demos have been improved to clearly mark required parameters.
- Replaced YOLOv5s-1 example json to YOLOv5s-6 json configuration file has been added for object detection.
- Documentation has been updated to add Python requirements and modified some images.
- feat: add OS and architecture checks in build script & update CPU specifications in documentation

### Fixed
- FPS calculation bug in yolo_multi
- Removed postprocessing code for legacy PPU models
- Fixed postprocessing logic to support new output shapes of YOLO models when USE_ORT=OFF
- fix typo error in framebuffer info file path (yolo_multi app)
- Improve error messages for output tensor size mismatch and missing in Yolo post processing
- Rename output tensors in json config 'yolov5s6_example.json'

### Added
- Added  to cleanly purge the pip-installed package and local build artifacts (shared library, dist-info/egg-info, and build directory).
- Added version guards in templates/python/yolo_pybind_example.py to ensure compatibility with DX-RT ≥ 3.0.0 and DXNN model version ≥ 7
- Enhanced the JSON configuration to support a target_output_tensor_name key and a name field for each layer parameter.
- Added a feature to filter output tensor using the target_output_tensor_name provided in the JSON configuration
- Added a feature to automatically reorder model layer parameters in the JSON configuration to match the model's actual output tensor sequence.
- Enhanced demo applications
- Added  for easy demo execution
- Added SCRFD decoding method for run_detector example
- Added postprocessing support for yolo_pose and yolo_face models (available only when USE_ORT=ON)
- Listed supported YOLO model types for YoloPostProcess in README
- feat: add uninstall script and enhance color utility functions

---

## DX-APP v1.11.0 / 2025-07-24

### Changed
- feat: enhance --clean option in build script for pybind artifacts
- feat: update dxnn models version(1.40.2 to 1.60.1)
- feat: auto run setup script or display a guide message when a file not found error occurs during example execution

### Fixed
- feat: Improve error message readability in install, build scripts
  - Apply color to error messages
  - Reorder message output to display errors before help messages
- Update tensor index assignment in Yolo layer reordering
- fix: resolve dx_postprocess Python lib build error and improve error handling

---

## DX-APP v1.10.0 / 2025-06-17

- Initial create dx-app   
- demo : classification    
- demo : object detection    
- demo : pose estimation    
- demo : multi models for object detection and segmentation    
- demo : semantic segmentation    
- demo : multi channel oject detection   
- template : classification     
- template : object detection    
- template : python example (sync/async/pybind c++)

---
