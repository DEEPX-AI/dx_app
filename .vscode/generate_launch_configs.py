#!/usr/bin/env python3
"""
Generate launch.json configurations for all examples

Usage:
    Generate launch.json file:
       python3 generate_launch_configs.py > launch.json
    
    Note: This will overwrite your existing launch.json file.
          Make sure to backup your launch.json if needed.
"""

import json

# Example configurations: (name, binary, model, video, image, rtsp_stream)
examples = [
    ("YOLOv5", "yolov5", "YOLOV5S_3.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv7", "yolov7", "YoloV7.dxnn", "snowboard.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv8", "yolov8", "YoloV8N.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv9", "yolov9", "YOLOV9S.dxnn", "carrierbag.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv10", "yolov10", "YOLOV10N-1.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv11", "yolov11", "YOLOV11N.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv12", "yolov12", "YOLOV12N-1.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv26", "yolov26", "YOLOV26S.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOX", "yolox", "YOLOX-S_1.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv5 Pose", "yolov5pose", "YOLOV5Pose640_1.dxnn", "dance-solo.mov", "sample/img/7.jpg", "stream6"),
    ("YOLOv8 Segmentation", "yolov8seg", "YOLOV8N_SEG-1.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("EfficientNet", "efficientnet", "EfficientNetB0_4.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("DeepLabV3", "deeplabv3", "DeepLabV3PlusMobileNetV2_2.dxnn", "blackbox-city-road.mp4", "sample/img/8.jpg", "stream6"),
    ("SCRFD PPU", "scrfd_ppu", "SCRFD500M_PPU.dxnn", "dance-group.mov", "sample/img/face_sample.jpg", "stream9"),
    ("YOLOv5 PPU", "yolov5_ppu", "YOLOV5S_PPU.dxnn", "boat.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv7 PPU", "yolov7_ppu", "YoloV7_PPU.dxnn", "snowboard.mp4", "sample/img/1.jpg", "stream6"),
    ("YOLOv5 Pose PPU", "yolov5pose_ppu", "YOLOV5Pose_PPU.dxnn", "dance-solo.mov", "sample/img/7.jpg", "stream6"),
]

# Multi-model example configurations: (name, binary, model1, model2, video, image, rtsp_stream)
multi_model_examples = [
    ("YOLOv7 x DeepLabV3", "yolov7_x_deeplabv3", "YoloV7.dxnn", "DeepLabV3PlusMobileNetV2_2.dxnn", "blackbox-city-road2.mov", "sample/img/8.jpg", "stream6"),
]

def create_config(name, binary, mode, input_type, model, video, image, rtsp_stream):
    """Create a single launch configuration"""
    
    # Determine arguments based on input type
    if input_type == "Image":
        args = ["-m", f"assets/models/{model}", "-i", image, "-l", "30"]
    elif input_type == "Image Dir":
        args = ["-m", f"assets/models/{model}", "-i", "sample/img", "-l", "30"]
    elif input_type == "Video":
        args = ["-m", f"assets/models/{model}", "-v", f"assets/videos/{video}"]
    elif input_type == "Video + Save":
        args = ["-m", f"assets/models/{model}", "-v", f"assets/videos/{video}", "-s"]
    elif input_type == "RTSP":
        args = ["-m", f"assets/models/{model}", "-r", f"rtsp://192.168.30.100:8554/{rtsp_stream}"]
    elif input_type == "Camera":
        args = ["-m", f"assets/models/{model}", "-c", "0"]
    else:
        args = []
    
    config = {
        "type": "cppdbg",
        "request": "launch",
        "name": f"Demo: {name} {mode} ({input_type})",
        "program": "${workspaceFolder}/bin/" + binary + "_" + mode.lower(),
        "args": args,
        "cwd": "${workspaceFolder}",
        "stopAtEntry": False,
        "environment": [],
        "externalConsole": False,
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": True
            }
        ],
        "preLaunchTask": f"build: {binary}_{mode.lower()}"
    }
    
    return config

def create_multi_model_config(name, binary, mode, input_type, model1, model2, video, image, rtsp_stream):
    """Create a single launch configuration for multi-model examples"""
    
    # Determine arguments based on input type
    if input_type == "Image":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-i", image, "-l", "30"]
    elif input_type == "Image Dir":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-i", "sample/img", "-l", "30"]
    elif input_type == "Video":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-v", f"assets/videos/{video}"]
    elif input_type == "Video + Save":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-v", f"assets/videos/{video}", "-s"]
    elif input_type == "RTSP":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-r", f"rtsp://192.168.30.100:8554/{rtsp_stream}"]
    elif input_type == "Camera":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-c", "0"]
    else:
        args = []
    
    config = {
        "type": "cppdbg",
        "request": "launch",
        "name": f"Demo: {name} {mode} ({input_type})",
        "program": "${workspaceFolder}/bin/" + binary + "_" + mode.lower(),
        "args": args,
        "cwd": "${workspaceFolder}",
        "stopAtEntry": False,
        "environment": [],
        "externalConsole": False,
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": True
            }
        ],
        "preLaunchTask": f"build: {binary}_{mode.lower()}"
    }
    
    return config

# Generate all configurations
all_configs = []

for name, binary, model, video, image, rtsp_stream in examples:
    for mode in ["Sync", "Async"]:
        for input_type in ["Image", "Image Dir", "Video", "Video + Save", "RTSP", "Camera"]:
            config = create_config(name, binary, mode, input_type, model, video, image, rtsp_stream)
            all_configs.append(config)

# Generate multi-model configurations
for name, binary, model1, model2, video, image, rtsp_stream in multi_model_examples:
    for mode in ["Sync", "Async"]:
        for input_type in ["Image", "Image Dir", "Video", "Video + Save", "RTSP", "Camera"]:
            config = create_multi_model_config(name, binary, mode, input_type, model1, model2, video, image, rtsp_stream)
            all_configs.append(config)

# Output JSON in launch.json format (pretty-printed)
launch_json = {
    "version": "0.2.0",
    "configurations": all_configs
}
print(json.dumps(launch_json, indent=4))
