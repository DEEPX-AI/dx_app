# Changelog 

## [v1.0.1] - 2024-08-08
### 1. Changed
- None
### 2. Fixed
- Fix a bug that occurred during the build with onnxruntime.
- modify install script to stabilize the installation.
### 3. Added
- None  

## [v1.0.0] - 2024-08-02
### 1. Changed
- dxnn version up(v6). so prior dxnn models will not work from this version.
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
- Fix a bug get_align_factor function that returned to the base value.
- Modify some functions to inline function call. It can Eliminating duplicate code and reducing overall code size. 
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
- Add library link syntax to enable onnxruntime c++ API.
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
   
