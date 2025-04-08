
**DX-APP** is DEEPX User's Application Templates based on DEEPX devices.    

This is an application examples that gives you a quick experience of NPU Accelerator performance.
You can refer to **DX-APP** and modify it a little or implement application depending on the purpose of your use.
This can reduce stress, such as setting the environment and implementing the code.
Application performance may also depending on the specifications of the host because it includes pre/post processing and graphics processing operations.

``` Note. Required libraries : OpenCV and dxrt. ```         

## System and Hardware Requirements
To Install DX Driver and Runtime, the following minimum requirements.   

**Operating System**   

  - Linux (Ubuntu 18.04 or later), aarch64 (20.04, 22.04)   
  - Windows 10 or later (64-bit)   

**Hardware**   

  - DX-M1 module connected via a Thunderbolt NVMe adapter   
  - M.2 connector with an M.2 board    

## Software Requirements
To run `dx_app`, you must install at least the following versions:   

  - Deepx M1 Firmware Version : **v1.6.3**   

**On Windows**

  - Deepx M1 Driver Version : **dxm1drv 11.34.15.609**
  - Deepx M1 Runtime Lib Version : **v2.8.0**

**On Linux**

  - Deepx M1 Driver Version : **v1.3.3**
  - Deepx M1 Runtime Lib Version : **v2.7.0** or later
        
## Overview

### Demos

**Classification**

This demo operates a classification models using images with a 224x224 input size. 
The output provides the **Top 1 Class** information. 

There are tow ways to view the NPU **Top 1**:

  - Directly checking the NPU Top 1 output(argmax_output, UINT16 Format).
  - Applying argmax on 1000 raw data points.

**Object Detection**

This demo operates object detection models with image, video, camera. 
When the input is in image format, the output is saved as "result.jpg", and the terminal displays the detected box information. 
When the input is video format (mp4, mov, avi … ), the output displays the video with detected boxes drawn on the screen.

**Object Det and Seg**

This demo uses two models to perform object detection and segmentation. 
When the input is image, the output is saved as “result.jpg”, and the terminal displays the detected. 
When the input video format, the output displays the video with detected boxes and drawn on the screen.   
```Note: This demo uses the following models – YOLOv5s_3.dxnn and DeepLabV3PlusMobileNetV2_2.dxnn.```

**Pose Estimation**

This demo operates pose estimation models with image, video, camera. 
It performs people and predicts joint point values for each detected object. 
When the demo runs, it displays a video with detection boxes and joint points drawn on the screen.    
```Note: This demo uses the following models – YOLOV5Pose640_1.dxnn```

### Templates

Application Template is that makes it easy to experience classification or object detection through json config file modification. 
The only part that needs to be modified is the json config file.

**Classifier**

For All classification models using image or binary files, You can obtain the Top 1 result with minor modifications to the JSON file. 
Please Refer to File example/imagenet_example.json for more details.

**Detector**

For some yolo series models using image, binary image, video, rtsp and camera. 
This is built using multi-threading programming. 
By continuously adding images or videos, it can display them as multi-channel outputs. 
Please Refer to Files example/yoloxxx_example.json for more details.