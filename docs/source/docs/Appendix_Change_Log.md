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
