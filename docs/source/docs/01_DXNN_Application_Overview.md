This chapter provides an overview of the DEEPX SDK architecture and explains each core component, and describes the overview and key features of DX-APP. 

## DEEPX SDK Architecture  
![](./../resources/01_SDK_Architecture.drawio_r2.png)

**DEEPX SDK** is an all-in-one software development platform that streamlines the process of compiling, optimizing, simulating, and deploying AI inference applications on DEEPX NPUs (Neural Processing Units). It provides a complete toolchain, from AI model creation to runtime deployment, optimized for edge and embedded systems, enabling developers to build high-performance AI applications with minimal effort.  

**DX-COM** is the compiler in the DEEPX SDK that converts a pre-trained ONNX model and its associated configuration JSON file into a hardware-optimized .dxnn binary for DEEPX NPUs. The ONNX file contains the model structure and weights, while the JSON file defines pre/post-processing settings and compilation parameters. DX-COM provides a fully compiled .dxnn file, optimized for low-latency and high-efficient inference on DEEPX NPU.  

**DX-RT** is the runtime software responsible for executing ,dxnn models on DEEPX NPU hardware. DX-RT directly interacts with the DEEPX NPU through firmware and device drivers, using PCIe interface for high-speed data transfer between the host and the NPU, and provides C/C++ and Python APIs for application-level inference control. DX-RT offers a complete runtime environment, including model loading, I/O buffer management, inference execution, and real-time hardware monitoring.  

**DX ModelZoo** is a curated collection of pre-trained neural network models optimized for DEEPX NPU, designed to simplify AI development for DEEPX users. It includes pre-trained ONNX models, configuration JSON files, and pre-compiled DXNN binaries, allowing developers to rapidly test and deploy applications. DX ModelZoo also provides benchmark tools for comparing the performance of quantized INT8 models on DEEPX NPU with full-precision FP32 models on CPU or GPU.  

**DX-STREAM** is a custom GStreamer plugin that enables real-time streaming data integration into AI inference applications on DEEPX NPU. It provides a modular pipeline framework with configurable elements for preprocessing, inference, and postprocessing, tailored to vision AI work. DX-Stream allows developers to build flexible, high-performance applications for use cases such as video analytics, smart cameras, and edge AI systems.  

**DX-APP** is a sample application that demonstrates how to run compiled models on actual DEEPX NPU using DX-RT. It includes ready-to-use code for common vision tasks such as object detection, face recognition, and image classification. DX-APP helps developers quickly set up the runtime environment and serves as a template for building and customizing their own AI applications.  

---

## DX-APP Features

**DX-APP** is a set of application templates designed to demonstrate the performance and deployment of AI models on DEEPX NPUs. It provides ready-to-use examples for image classification, object detection, segmentation, and pose estimation.

You can quickly evaluate inference capabilities without modifying the source code and easily adapt the templates for custom applications. These templates significantly reduce the overhead of environment configuration and manual implementation.

!!! note "NOTE" 

    Application performance may vary depending on host system specifications. Each demo includes pre-processing, post-processing, and graphics processing operations.


### Demos

**DX-APP** demos are optimized to showcase pre-compiled models on DEEPX NPUs with minimal setup. Each demo represents a common AI task and can be executed using images, videos, or live camera input.

**Classification**  

- Executes classification models with image inputs (e.g., `224x224`).  
- Outputs the Top-1 predicted class.  
- Example: `example/run_classifier/imagenet_example.json`

**Object Detection**  

This demo supports image, video (mp4, mov, avi), and camera input.  

- For image input, outputs result.jpg and prints detected objects to the terminal.  
- For video input, displays bounding boxes on the output video.  

**Pose Estimation**  

- Detects people and estimates keypoints (joints) using image, video, or camera input.  
- The output includes both bounding boxes and joint coordinates rendered on screen.  
  
**Segmentation**  

This demo uses two models to perform segmentation.   

- For image input, saves results to result.jpg and prints info to the terminal.  
- For video input, displays output with both detection boxes and segmentation masks.  


### Templates

**DX-APP** provides lightweight Application Templates that run classification or detection models by modifying only the JSON configuration. No code changes required.

**Classification**  

- Supports various classification models with image or binary input.  
- Outputs the Top-1 class by adjusting JSON fields.  
- Example JSON: `example/run_classifier/imagenet_example.json`  

**Object Detection**  

- Supports YOLO-series models using image, binary, video, RTSP stream, or camera input.  
- Built on multi-threading for multi-channel and real-time processing.  
- Supports dynamic input source expansion and grid-style output display.  
- Example JSON: `example/run_detector/yolov5s3_example.json`

---
