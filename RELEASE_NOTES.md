# RELEASE_NOTES
## v3.0.2 / 2026-02-10

### 1. Changed
- Copy of dxrt and ckpkg DLLs into the dx-app/bin directory when building with MSVC.

### 2. Fixed
- Remove experimental filesystem includes and update float literals in example cpp files for build error on windows

### 3. Added
- Added vcpkg installation script for windows build. 

## v3.0.1 / 2026-02-05

### 1. Changed

### 2. Fixed
- Hardcoded attribute size in YOLO post-processing to dynamically adjust based on model output shape

### 3. Added
- Add yolov26 cls, yolo26 pose, yolo26 seg, yolo26 obb examples

## DX-APP v3.0.0 / 2026-01-02

### 1. Changed

#### 1.1 Major Project Structure Refactoring
- **Complete overhaul from existing demo applications to example system**
  - To improve user understanding, separated the previously integrated example code by Task (classification, object detection, segmentation, face recognition, pose estimation) / Model (EfficientNet, YOLO, YOLO_PPU, SCRFD, ...) / Inference method (sync, async) / Post-processing (pure python, pybind)
    - Complete removal of legacy C++ demo code in `demos/` directory and provision of `run_demo.sh` and `run_demo.bat` based on separated examples
    - Transition to new `src/cpp_example/` and `src/python_example/` structure

#### 1.2 Build System Improvements
- Improved CMake configuration and enhanced shared library support
- Updated C++17 and Visual Studio 2022(v143) configuration for Windows build
- Adjusted DXRT include and link directories for cross-compilation

#### 1.3 Complete Reconstruction of C++ / Python Example System
- **Support for synchronous and asynchronous execution modes**
- **Support for various input sources**: image, video, camera, RTSP stream
- **Real-time processing mode**: Performance measurement without GUI using `--no-display` option
- **Enhanced performance profiling**: 
  - Latency measurement for each stage: preprocessing, inference, post-processing
  - E2E(End-to-End) FPS calculation and performance report generation
  - Automatic generation of timestamp-based performance report files

#### 1.4 Model Support Expansion
- **YOLOv10, YOLOv11, YOLOv12** examples added
- **YOLOv8 Segmentation** (YOLOv8-seg) support
- **DeepLabv3** segmentation model support
- **PPU (Post-Processing Unit)** module integration:
  - YOLOv5, YOLOv7 PPU version support
  - SCRFD PPU version support
  - Both Python and C++ examples provided

#### 1.5 Documentation Improvements
- Newly written example guides and installation guides
- Added detailed usage examples and parameter descriptions for each model

### 2. Fixed

#### 2.1 Code Quality Improvements
- Added try-catch error handling to all projects
- Improved `std::exception` handling and throw `std::invalid_argument` when layer requirements are not met
- Removed `using namespace std` usage and improved code clarity with explicit `std::` usage
- Improved parameter handling and frame processing logic
- Enhanced argument validation and error messages

#### 2.2 Input Processing Improvements
- Set `cv2.CAP_PROP_BUFFERSIZE` (buffer size 1) for camera and RTSP speed improvement
- Fixed input_tensor passing to maintain memory reference until asynchronous inference completes

### 3. Added

#### 3.1 Post-processing Library (dx_postprocess)
- **Pybind11-based Python binding**:
  - Provides Python binding for C++ post-processing functions
  - Automatically installs to current Python execution environment

#### 3.2 Test Infrastructure Construction
- **Pytest-based integrated test system**:
  - Automated testing for all Python examples
  - Achieved code coverage of 93.65% or higher
  - E2E(End-to-End) test framework
  - Includes all model tests for classification, object detection, segmentation, and pose estimation
- Added `.coveragerc` file for code coverage configuration
- Support for display mode and E2E mode testing

#### 3.3 New Examples and Features
- **Classification Models**:
  - EfficientNet example integration
  - ImageNet classification examples (synchronous/asynchronous)
  
- **Object Detection Models**:
  - YOLO26 (Support new Ultralytics model, which is optimized for edge deployment)
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

### 4. Removed or Replaced

#### 4.1 Legacy Demo Removal
- Complete removal of `demos/classification/`
- Complete removal of `demos/object_detection/`
- Complete removal of `demos/segmentation/`
- Complete removal of `demos/pose_estimation/`
- Complete removal of `demos/face_recognition/`
- Removal of `demos/denoiser/`
- Removal of `demos/dncnn_yolo/`
- Removal of `demos/object_det_and_seg/`
- Removal of `demos/noiseVideoMaker/`

#### 4.2 Legacy Configuration File Removal
- Complete removal of JSON configuration files in `example/` directory
- Removal of `example/dx_postprocess/` JSON files
- Removal of Debian package related files (`debian/`)
- Removal of Docker build files (`docker/Dockerfile.app.build`)

#### 4.3 Legacy Code Cleanup
- Removal of `demo_utils/` directory
- Removal of duplicate or unused code
- Removal of old YOLOv5 post-processing files
- Removal of RISCV64 architecture support

### 5. Migration Guide

#### 5.1 Notice for Existing Users
v3.0.0 is a major update that includes **Breaking Changes** compared to v2.x.

- **Demo Code**: The existing `demos/` directory has been completely removed. Please refer to the new examples in `src/cpp_example/` and `src/python_example/`.
- **JSON Configuration Files**: JSON files in the existing `example/` directory have been removed. Python examples are configured directly through command-line arguments.
- **YOLO Post-processing Type Names**: Some have been changed, but aliases are provided for backward compatibility.

#### 5.2 Recommended Upgrade Path
1. Refer to Python example documentation
2. Check example code in `src/python_example/` or `src/cpp_example/` directory
3. Install Python dependencies through `requirements.txt`
4. Use build scripts (`build.sh` or `build.bat`)

### 6. Known Issues
- When using the PPU model for face detection & pose estimation, `dx-compiler v2.1.0 and v2.2.0` does not currently support converting face and pose models to PPU format. This feature will be added in a future release. The PPU models used in the demo were converted using dx-compiler `v1.0.0(dx_com v1.60.1)`.

---

## DX-APP v2.1.0 / 2025-11-28

### 1. Changed
- Enhance build script documentation and usage instructions
- Update cmake configuration in build.bat to use C++17 and v143 for enhance documentation windows build script(visual studio 2022)
- Model package updated from version 2.0.0 to 2.1.0 to support PPU models
- Improved demo script with additional PPU-Demo (1, 4, 6, 8, 11)
- Added CPU-specific PyTorch wheel source (https://download.pytorch.org/whl/cpu) in templates/python/requirements.txt.

### 2. Fixed
- Fix Windows MSBuild compilation warnings by replacing implicit type casts with explicit static_cast
- Improve tensor allocation in imagenet classification example
- Update numBoxes calculation based on post-processing type in LayerReorder
- Rename YOLO post-processing types and add aliasing for backward compatibility
- Add VSCode configuration files for usability
- Fixed errors that occurred when using VAAPI with camera input
- Enhanced yolo application to display final FPS even when forcefully terminated during camera input usage
- Enhance user input handling for run_demo selection with a countdown timer (20s)

### 3. Added
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

### 4. Known Issues
- DeepLabV3 Semantic Segmentation model accuracy may be slightly degraded in dx-compiler(dx_com) v2.1.0. This will be fixed in the next release. The DeepLabV3 model used in the demo was converted using dx-compiler v2.0.0.
- When using the PPU model for face detection & pose estimation, dx-compiler v2.1.0 does not currently support converting face and pose models to PPU format. This feature will be added in a future release. The PPU models used in the demo were converted using dx-compiler v1.0.0(dx_com v1.60.1).

---

## DX-APP v2.0.0 / 2025-08-14

### 1. Changed
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

### 2. Fixed
- FPS calculation bug in yolo_multi
- Removed postprocessing code for legacy PPU models
- Fixed postprocessing logic to support new output shapes of YOLO models when USE_ORT=OFF
- fix typo error in framebuffer info file path (yolo_multi app)
- Improve error messages for output tensor size mismatch and missing in Yolo post processing
- Rename output tensors in json config 'yolov5s6_example.json'

### 3. Added
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
### 1. Changed
- feat: enhance --clean option in build script for pybind artifacts
- feat: update dxnn models version(1.40.2 to 1.60.1)
- feat: auto run setup script or display a guide message when a file not found error occurs during example execution

### 2. Fixed
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
