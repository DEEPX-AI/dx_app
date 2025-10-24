## Version 2.1.0 (Sep 2025)
- Windows Environment Support
DX-APP now fully supports the Windows operating system! In response to user requests, we've expanded compatibility beyond Linux to include Windows, enabling a broader range of development environments to take advantage of DX-APP.
  - **OS**: Windows 10 / 11
  - **Deepx M1 Driver Version**: v1.7.1 or higher
  - **Deepx M1 Runtime Lib Version**: v3.1.0 or higher
  - **Python**: Version 3.8 or higher (required for Python module support)
  - **Compiler**: Visual Studio Community 2022 (required for building C++ examples)
- Fix Windows MSBuild compilation warnings by replacing implicit type casts with explicit static_cast
- Add automated build script (build.bat) for automatic build and Visual Studio solution generation
- Enhance build script documentation and usage instructions
- Update CMake configuration in build.bat to use C++17 and enhance documentation for build script
- Added CPU-specific PyTorch wheel source (https://download.pytorch.org/whl/cpu) in templates/python/requirements.txt.
- Fixed errors that occurred when using VAAPI with camera input
- Enhanced yolo application to display final FPS even when forcefully terminated during camera input usage
- Three new PPU data types : BBOX (for object detection) / POSE (for pose estimation keypoints) / FACE (for face detection landmarks)
  - Handles bounding box decoding and score thresholding functions directly on the NPU hardware instead of the CPU.
  - Key Benefits of PPU : Particularly beneficial for real-time applications requiring high frame rates
  - Model package updated from version 2.0.0 to 2.1.0 to support PPU models
  - Improved demo script with additional PPU-Demo (3, 5, 7, 10)

## Version 2.0.0 (August 2025)
- Major code refactoring and restructuring of demo applications.
- Update on Tensor Index Assignment in Yolo Layer Reordering
- demos/demo_utils/yolo_cfg.cpp Structure Changes: To accommodate different post-processing methods based on the USE_ORT setting in RT, the final output name of the ONNX model is now received as a separate parameter.
- run_detector Updates: For more efficient post-processing, run_detector also now receives the final ONNX output name as a separate parameter.
- yolo_pybind_example.py Refactoring: Refactored the code to use a RunAsync() + Wait() structure instead of callbacks. This change ensures the correct handling of the output tensor order.
- Improved FPS calculation for the YOLO multi-demo.

## Version 1.10.0 (June 2025)
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
