# RELEASE_NOTES
## v2.1.0 / 2025-09-18

### 1. Changed
- Enhance build script documentation and usage instructions
- Update cmake configuration in build.bat to use C++17 and v143 for enhance documentation windows build script(visual studio 2022)

### 2. Fixed
- Fix Windows MSBuild compilation warnings by replacing implicit type casts with explicit static_cast
- Improve tensor allocation in imagenet classification example
- Update numBoxes calculation based on post-processing type in LayerReorder
- Rename YOLO post-processing types and add aliasing for backward compatibility
- Add VSCode configuration files for usability

### 3. Added
- Windows Environment Support
DX-APP now fully supports the Windows operating system! In response to user requests, we've expanded compatibility beyond macOS and Linux to include Windows, enabling a broader range of development environments to take advantage of DX-APP.
  - **OS**: Windows 10 / 11
  - **Deepx M1 Driver Version**: v1.7.1 or higher
  - **Deepx M1 Runtime Lib Version**: v3.1.0 or higher
  - **Python**: Version 3.8 or higher (required for Python module support)
  - **Compiler**: Visual Studio Community 2022 (required for building C++ examples)
- Add automated build script (build.bat) for automatic build and Visual Studio solution generation

## v2.0.0 / 2025-08-14

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

## v1.11.0 / 2025-07-24

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

### 3. Added

## v1.10.2 / 2025-07-03
### 1. Changed
### 2. Fixed
- handle runtime error when VAAPI plugin is installed but element is unavailable
### 3. Added

## v1.10.1 / 2025-06-23
### 1. Changed
### 2. Fixed
- fix: fix dx-rt build error caused by pybind11 incompatibility with Python 3.6.9 on Ubuntu 18.04
  - Support automatic installation of minimum required Python version (>= 3.8.2)
  - Install Python 3.8.2 if the system Python version is not supported
  - On Ubuntu 18.04, install via source build; on Ubuntu 20.04+, use apt install
  - Added support in install.sh to optionally accept --python_version and --venv_path for installation
  - Added support in build.sh to accept and use --python_exec
  - Added support in build.sh to optionally accept --venv_path and activate the specified virtual environment
### 3. Added

## v1.10.0 / 2025-06-17
### 1. Changed
- Improved FPS display accuracy in the yolo_multi demo by average over the last {window_size} seconds. (add window_size parameter)
- Added DeepX developer page URL in MkDocs configuration.
- Updated the appendix page to include the current version and initial release information.
### 2. Fixed
### 3. Added

## v1.9.9 / 2025-06-13
### 1. Changed
### 2. Fixed
- fix: prevent overwriting result files by using unique names in Python example
- chore: update Python dependencies to use CPU version of torch
### 3. Added

## v1.9.8 / 2025-06-09
### 1. Changed
### 2. Fixed
- update condition for FPS calculation
### 3. Added

## v1.9.7 / 2025-06-09
### 1. Changed
### 2. Fixed
- refactor FPS calculation variables for clarity in yolo_multi
### 3. Added

## v1.9.6 / 2025-06-09
### 1. Changed
- Improved the installation method of the Python-based dx_postprocess module to use pip install .
- optimize processing time calculation and improve frame handling in object detection demo
### 2. Fixed
- Added handling for empty frame indexing queue
### 3. Added
- run / run_batch options related to the ImageNet Python example
- Automatically downloads pybind11 via git clone in setup.py (header-only, no build required).


## v1.9.5 / 2025-06-05
### 1. Changed
- Updated imageNet_example.py to use run_batch
- Changed imagenet_classification C++ demo to run in an infinite loop
- Updated documentation to reflect new file structure
### 2. Fixed
- None
### 3. Added
- None

## v1.9.4 / 2025-06-02
### 1. Changed
- Updated camera backend selection logic — now uses:
      windows : auto
      linux : v4l2
- drop wait mode support in template/python/yolo_async.py and remove callback_mode and wait option parameters
### 2. Fixed
- Corrected the FPS calculation logic by using a cumulative average instead of per-frame timing.
### 3. Added
- None

## v1.9.3 / 2025-05-29
### 1. Changed
- None
### 2. Fixed
- Fix a correct JSON file path in script
- update DX-RT Python API to handle RT version incompatibility
### 3. Added
- None

## v1.9.1 / 2025-05-16
### 1. Changed
- Moved all exposed Windows .bat files into the x86_64_win directory for better organization.
### 2. Fixed
- Removed unused tools and the profiler directory to clean up the project structure.
### 3. Added
- None

## v1.9.0 / 2025-04-17
### 1. Changed
- restructure JSON files under example directory (grouped by app name under ./bin)
- clean up *example.json files (remove unused properties and maintain consistency)
### 2. Fixed
- correct pybind API module installation path and update example JSON
### 3. Added
- add YOLOv5 face model example to 'run_detector' app

## v1.8.1 / 2025-04-15
### 1. Changed
- None
### 2. Fixed
- Fixed an issue where sigmoid was applied twice in the Python example when using the yolov5s model with all_decode mode.
### 3. Added
- add support for MSVC Debug mode build on Windows
- None

## v1.8.0 / 2025-04-14
### 1. Changed
- The dx_postprocess module is now installed directly into the current Python interpreter’s site-packages during build.
- Output bounding box coordinates from post-processing are now always expressed in original image coordinates.
### 2. Fixed
- When running inference with a PPU model in Python, a batch size dimension is added to the output tensor’s shape. The dx_postprocess module was updated to handle this change properly during post-processing.
### 3. Added
- Added SetConfig(py::dict config) method to the YoloPostProcess class to allow updating post-processing parameters at runtime.

## v1.7.2 / 2025-04-03
### 1. Changed
- Reorganized each folder's README.md content using MkDocs.
- Removed README.md files and created corresponding docs/ directories.
- None
### 2. Fixed
- Support Ubuntu 24.04 environment for 'install.sh'
### 3. Added
- None

## v1.7.1 / 2025-03-27
### 1. Changed
- None
### 2. Fixed
- Fixed post-processing bugs in DeepLabV3 Segmentation demo (demos/segmentation) and Classification demo (demos/classification) when USE_ORT=ON.
### 3. Added
- Added setup.bat to retrieve assets via S3 on Windows OS.

## v1.7.0 / 2025-03-24
### 1. Changed
- Updated UI for the yolo_multi display window.
- Download assets(model, video) from AWS S3
- Change assets model path and rename reference JSON file.
### 2. Fixed
- Modify to enable GPU acceleration API (gstreamer vaapi) when using video files.
- Fix bug in align dummy size calculation in the demos/classification/classification.cpp
- Modify the installation path to ensure the dx_postprocess Python API works in both container and Windows environments.
- None
### 3. Added
- Support post-processing for the provided YoloV8N with USE_ORT=OFF version.
- None

## v1.6.0 / 2025-03-19
### 1. Changed
- None
### 2. Fixed
- None
### 3. Added
- Support Intel GPU HW acceleration

## v1.5.0 / 2025-02-28
### 1. Changed
- Refactored project for MSVC build compatibility
- Improved code compatibility between Windows and Linux by using '#if - #else' directives 
### 2. Fixed
- None
### 3. Added
- Added batch files for application executions in Windows
- Included 'CMakeSettings.txt' and 'vcpkg.json' for MSVC build support

## [v1.4.1] - 2025-02-26
### 1. Changed
- None
### 2. Fixed
- Fixed a "Segmentation Fault" issue in a specific situation caused by accessing a vector out of its size range during post-processing..
### 3. Added
- Added parsing of "dxrt-cli -s" output with the "yolo_multi -t" option to monitor NPU temperature, logging the information and saving it to a file.

## [v1.4.0] - 2025-02-13
### 1. Changed
- Updatetd the post-processing code to use `NumPy` instead of `torch.max()` in `template/python/yolov5s_example.py` to reduce latency.
- Updatetd the post-processing code to use `NumPy` instead of `torch.max()` in `template/python/yolo_async.py` to reduce latency.
### 2. Fixed
- Fixed the hardcoded number of classes in `template/python/yolov5s_example.py`.
### 3. Added
- Added `dx_postprocess` pybind module containing `YoloPostProcess` class.
- Added `template/python/yolo_pybind_example.py` as an example of how to use the `YoloPostProcess` class.
- Added `template/python/README.md` with explainations of the Python examples.

## [v1.3.0] - 2025-01-09
### 1. Changed
- Apply new feature API from dx_rt; May lead to perfomance degradation when using older versions of dx_app.
- Update install.sh to support manual installation of OpenCV via source build or automatic installation using apt-get.
### 2. Fixed
- None
### 3. Added
- Add 3-second timeout for camera connection test. (template/object_detection/run_detector)

## [v1.2.2] - 2025-01-09
### 1. Changed
- None
### 2. Fixed
- Modify yolov9 decode method in config json file. 
### 3. Added
- Add yolov9 example dxnn model

## [v1.2.1] - 2024-12-20
### 1. Changed
- None
### 2. Fixed
- Fix a bug yolo post-processing in python codes and simplified.
### 3. Added
- None

## [v1.2.0] - 2024-12-12
### 1. Changed
- Fixed Python code due to changes in the DXRT Python API. Use a specific version of DXRT(2.6.1).
### 2. Fixed
- Fix a bug when using USE_ORT=ON, Some demo codes may or may not work very well.
### 3. Added
- None

## [v1.1.3] - 2024-12-12
### 1. Changed
- None
### 2. Fixed
- update scripts for install opencv (in OP5-Plus 22.04)
### 3. Added
- Add yolo_async.py, that Implemented an runAsync method for object detection demo using YOLO model.

## [v1.1.2] - 2024-11-14
### 1. Changed
- None
### 2. Fixed
- Remove the ONNX Runtime linking process from the build, eliminating the "USE_ORT" conditional check.
### 3. Added
- Add age and gender classification logic from the demo/face_recognition application.

## [v1.1.1] - 2024-10-29
### 1. Changed
- None
### 2. Fixed
- Install opencv Check if the required version(4.5.5) of OpenCV is installed in install script.
### 3. Added
- None

## [v1.1.0] - 2024-09-30
### 1. Changed
- Set as the default Async Mode for "yolo_multi" and "run_detector"
### 2. Fixed
- None
### 3. Added
- Add "yolov8" decoding method (It must be used with `USE_ORT=ON`)

## [v1.0.7] - 2024-09-27
### 1. Changed
- None
### 2. Fixed
- Fix a bug that cross compile build error with onnxruntime library
### 3. Added
- None

## [v1.0.6] - 2024-09-20
### 1. Changed
- Modify "ethernet" option to "rtsp"
### 2. Fixed
- Fix a bug that de-noiser post-processing pitch size (64 to output channel size)
### 3. Added
- Add a README.md files for guide customizing post processing and explanation templates working

## [v1.0.5] - 2024-09-11
### 1. Changed
- Update example models new version
### 2. Fixed
- Fix a bug RTSP mode in demos/yolo_demo.cpp, add parameter "-r" and "--rtsp"
- Fix a bug that overlapping face ID in face recognition demo
### 3. Added
- None

## [v1.0.4] - 2024-09-06
### 1. Changed
- None
### 2. Fixed
- Fix a bug in object detection example of NMS calculation error in python file
- Fix a bug in SCRFD Post Processing, (ll.283 ~ 290, in yolo_post_processing.hpp)
### 3. Added
- None

## [v1.0.3] - 2024-09-04
### 1. Changed
- Delete cpu onnx file
- Yolo configuration parameter was changed(Delete ppu and concat yolo config), Please check yolo_demo code or README.md
### 2. Fixed
- Added output node name to YoloLayerParam for layer reordering (demo/object_detection)
- Fix a bug with python example code
### 3. Added
- None

## [v1.0.2] - 2024-08-22
### 1. Changed
- None
### 2. Fixed
- Fix output layers order distortion problem
### 3. Added
- Supporting RTSP URL input for yolo & yolo_multi & run_detector application

## [v1.0.1] - 2024-08-08
### 1. Changed
- None
### 2. Fixed
- Fix a bug that occurred during the build with onnxruntime
- modify install script to stabilize the installation
### 3. Added
- None  

## [v1.0.0] - 2024-08-02
### 1. Changed
- dxnn version up(v6). so prior dxnn models will not work from this version
- Update to dxnn model file version 6
- Update imagenet example python code for none argmax model
### 2. Fixed
- Fix a install script syntex error
### 3. Added
- None  

## [v0.2.4] - 2024-07-22
### 1. Changed
- Update models for new architecture
### 2. Fixed
- stabilizing python example codes
### 3. Added
- None 

## [v0.2.3] - 2024-07-17
### 1. Changed
- Update post processing methods for new architecture
### 2. Fixed
- None
### 3. Added
- None 

## [v0.2.2] - 2024-06-18
### 1. Changed
- None
### 2. Fixed
- Fix a bug yolo multi demo video loop mode
### 3. Added
- None 

## [v0.2.1] - 2024-06-13
### 1. Changed
- None
### 2. Fixed
- Fix a bug yolo demo video loop mode
### 3. Added
- None 

## [v0.2.0] - 2024-06-11  
### 1. Changed
- None 
### 2. Fixed
- None
### 3. Added
- Update usage of custom decode function
- Add yolopose decode function in template

## [v0.1.5] - 2024-06-04  
### 1. Changed
- None
### 2. Fixed
- Fix a bug yolo model config
- Update example model version 3 to 4
### 3. Added
- None 

## [v0.1.4] - 2024-06-03  
### 1. Changed
- None
### 2. Fixed
- Fix a bug get_align_factor function that returned to the base value
- Modify some functions to inline function call. It can Eliminating duplicate code and reducing overall code size
- Modify supported architecture name arm64 to aarch64
### 3. Added
- None 

## [v0.1.3] - 2024-05-28  
### 1. Changed
- None
### 2. Fixed
- Fix a bug in the script file run_xx.sh
  Remove configs folder (Duplicated meaning of "configs" and "example" folders)
### 3. Added
- None 

## [v0.1.2] - 2024-05-24  
### 1. Changed
- Add library link syntax to enable onnxruntime c++ API
### 2. Fixed
- None
### 3. Added
- None 

## [v0.1.1] - 2024-05-20  
### 1. Changed
- None
### 2. Fixed
- Support gnu 11.4 compiler for ubuntu 22.04
### 3. Added
- None 

## [v0.1.0] - 2024-04-29  
### 1. Changed
- None
### 2. Fixed
- None
### 3. Added
- Initial create dx-app   
- demo : classification    
- demo : object detection    
- demo : pose estimation    
- demo : multi models for object detection and segmentation    
- demo : semantic segmentation    
- template : classification     
- template : object detection    
- template : python example (classification)    
   
