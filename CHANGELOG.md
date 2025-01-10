# Changelog 

## [v1.3.0] - 2025-01-09
### 1. Changed
- Apply new feature API from dx_rt; May lead to perfomance degradation when using older versions of dx_app.
### 2. Fixed
- None
### 3. Added
- None

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
   
