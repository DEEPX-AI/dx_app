This chapter introduces a quick-start experience using pre-built demo applications provided by **DX-APP**. These applications allow developers to evaluate DeepX NPU performance on common vision AI tasks such as classification, detection, and segmentation. Developers can modify the examples or use them as templates to build custom applications.  

**Note.** Performance results may vary depending on host system specifications, as the demos include host-side pre-processing, post-processing, and rendering operations.  

---

## C++ Demo Application List  

**DX-APP** provides the following C++ demo applications.  

- **Classification**: Basic Classification, ImageNet Classification  
- **Object Detection**: Yolo Object Detection, Yolo Object Detection - Multi Channel  
- **Pose Estimation**: Human Pose Estimation  
- **Segmentation**: Semantic Segmentation DeepLabV3 (CityScape dataset Only), Semantic Segmentation DeepLabV3 (CityScape dataset Only) + Yolo Object Detection  

---

## Running Demo Executables  

Each demo can be executed on Linux or Windows.  

### Classification  

This section demonstrates how to run the ImageNet Classification demo on Windows.  

- Model: ImageNet-trained `EfficientNetB0_4.dxnn` model  
- Input: Single image file  
- Output: Top-1 classification result printed to the terminal  

**Classification Demo**  

- Linux Command  

```
./bin/classification -m assets/models/EfficientNetB0_4.dxnn -i sample/ILSVRC2012/1.jpeg
```

- Windows Command  

```
bin\classification.exe -m assets/models/EfficientNetB0_4.dxnn -i sample/ILSVRC2012/1.jpeg
```

- Output Example  

```
Top1 Result : class 321
[DXAPP] [INFO] total time : 5703 us
[DXAPP] [INFO] per frame time : 5703 us
[DXAPP] [INFO] fps : 200
```

**ImageNet Classification Demo**  

- Linux Command  

```
./bin/imagenet_classification -m assets/models/EfficientNetB0_4.dxnn -i example/imagenet_classification/imagenet_val_map.txt -p sample/ILSVRC2012/
```

- Windows Command  

```
bin\imagenet_classification.exe -m assets\models\EfficientNetB0_4.dxnn -i example\imagenet_classification\imagenet_val_map.txt -p sample\ILSVRC2012\
```

- Output Example  

![](./../resources/03_01_Output_of_imagenet.png){ width=400px }

The output shows the accuracy of the classification result is **74.3%** and the frame rate (fps) is **634**.  


### Object Detection  

This section explains how to run object detection demos using YOLOv5 models. Both single-stream and multi-stream inference scenarios are supported.  

- Model: `yolov5s_512`  

**YOLO Object Detection - Single Channel**  
This demo performs object detection on a single input stream using a YOLOv5 model.  

- Output Example: Upon execution, the application displays detected objects.  

```
./bin/yolo -m assets/models/YOLOV5S_3.dxnn -i sample/1.jpg -p 1
...
  Detected 10 boxes.
    BBOX:person(0) 0.877039, (308.116, 138.729, 400.442, 363.999)
    BBOX:bowl(45) 0.760544, (25.5619, 359.37, 78.8973, 393.056)
    BBOX:bowl(45) 0.749591, (46.0043, 315.064, 107.362, 346.737)
    BBOX:oven(69) 0.706145, (0.168434, 228.332, 154.418, 324.19)
    BBOX:person(0) 0.631538, (0.423515, 295.068, 47.6304, 332.823)
    BBOX:bowl(45) 0.586059, (0.148106, 329.093, 68.946, 379.93)
    BBOX:oven(69) 0.571996, (389.631, 245.565, 495.666, 359.34)
    BBOX:bottle(39) 0.455668, (172.258, 268.919, 200.505, 322.398)
    BBOX:pottedplant(58) 0.442108, (0.457752, 86.3962, 51.3189, 208.127)
    BBOX:bowl(45) 0.407671, (124.369, 219.818, 145.423, 232.568)
  Result saved to result.jpg
  [DXAPP] [INFO] total time : 23618 us
  [DXAPP] [INFO] per frame time : 23618 us
  [DXAPP] [INFO] fps : 43.4783
```

In this example, a person is detected with **confidence 0.877**, and the bounding box is defined by the four coordinates.

![](./../resources/03_02_Output_of_yolo.png){ width=600px }

**Pre-processing and Post-processing Parameters**  
YOLO models in **DX-APP** require external configuration for pre-processing and post-processing parameters. These parameters are not embedded in the `.dxnn` model.file.  

Pre-processing and post-processing parameters  

- Defined in:  `demos/demo_utils/yolo_cfg.cpp`  
- Referenced in: `yolo_1channel.cpp`  
 
To customize,  

- Modify the existing parameters in both files, or   
- Add new parameter entries with a new model name to both files.  

```
YoloParam yolov5s_512 = {
    512,  // height
    512,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    0,   // numBoxes
    80,   // numClasses
    "output", // onnx output name
    {     // if use_ort = off, layers config
        createYoloLayerParam("378", 40, 40, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("439", 20, 20, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("500", 10, 10, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
        .
        .
        .
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle", . . .}
}
```

**YOLO Object Detection - Multi Channel**  
This demo performs object detection on multiple input streams simultaneously using a YOLOv5 model. 

- Output Example: Upon execution, detection results across all input streams will be displayed in a tiled or multi-channel format.  

```
./bin/yolo_multi -c ./example/yolo_multi/yolo_multi_demo.json

```

![](./../resources/03_03_Output_of_yolo_multi.png){ width=700px }


**JSON Configuration for Multi-Channel YOLO Demo**  
The multi-channel demo uses a JSON configuration file to to define the input sources and model behavior.  

- File Location: `example/yolo_multi/yolo_multi_input_source_demo.json`  

Edit this file to customize the input video sources and number of streams.  

**Offline Mode in Input Method**  
To run detection in offline mode (for video file)  

- Set the `video_sources` field in the JSON file  
- Specify the number of frames to process as the third parameter  

```
{
    "usage": "multi",
        "model_path": "/model_path/yolov5s_512", "model_name": "yolov5s_512",

        "video_sources": [
            ["./assets/videos/dron-citry-road.mov", "realtime"], 
            ["/dev/video0", "camera"],
            ["./sample/1.jpg", "image"], 
            ["rtsp://210.99.70.120:1935/live/cctv010.stream", "rtsp"], 
            ["./assets/videos/dance-group.mov", "offline", 400]
        ],

        "display_config": {
            "display_label": "output", 
            "capture_period": 33,
            "output_width": 1920,
            "output_height": 1080, 
            "show_fps": true
    }
}
```

**Pre-processing and Post-processing Parameters**  
YOLO models in **DX-APP** require external configuration for pre-processing and post-processing parameters. These parameters are **not** embedded in the compiled `.dxnn` model file.  

Pre-processing and post-processing parameters  

- Defined in:  `demos/demo_utils/yolo_cfg.cpp`  
- Referenced in: `yolo_multi_channels.cpp`  

To customize,  

- Modify the existing parameters in both files, or  
- Add new parameter entries with a new model name to both files.  

```
YoloParam yolov5s_512 = {
    512,  // height
    512,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    0,   // numBoxes
    80,   // numClasses
    "output", // onnx output name
    {     // if use_ort = off, layers config
        createYoloLayerParam("378", 40, 40, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("439", 20, 20, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("500", 10, 10, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
        .
        .
        .
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle", . . .}
}
```

```
YoloParam getYoloParameter(string model_name){ 
    if(model_name == "yolov5s_320")
        return yolov5s_320;
    else if(model_name == "yolov5s_512") 
        return yolov5s_512;
    else if(model_name == "yolov5s_640") 
        return yolov5s_640;
    else if(model_name == "yolox_s_512") 
        return yolox_s_512;
    else if(model_name == "yolov7_640") 
        return yolov7_640;
    else if(model_name == "yolov7_512") 
        return yolov7_512;
    else if(model_name == "yolov4_608") 
        return yolov4_608;
    return yolov5s_512;
}
```

**RTSP Stream Input**  
To run a demo using an **RTSP** video stream, configure the input source in the JSON file as follows.  

- Set the **RTSP** stream address under the `video_sources` field  
- Set the network type to `rstp`  

```
{
    .
    .
    .
    "video_sources": [
        ["rtsp://your_rtsp_stream_address", "rtsp"]
    ],
    .
    .
    .
}
```

This enables real-time inference directly from network video streams using DEEPX NPU.  


### Pose Estimation  

This section explains how to run pose estimation demos based on Ultralytics YOLO Pose model using DEEPX NPU.  

- Model: Ultralytics YOLO Pose (`YOLOV5Pose640_1.dxnn`)  

**Pose Estimation**
```
./bin/pose -m assets/models/YOLOV5Pose640_1.dxnn -i sample/7.jpg -p 0
```

The YOLO pose model predicts human key points for each detected person within an image or video frame.  

![](./../resources/03_04_Output_of_YOLO_Pose_Estimation.png){ width=600px }

**Pre-processing and Post-processing Parameters**  
Yolo Pose models do not embed external pre-processing and post-processing parameters in the compiled `.dxnn` file.  

- Parameters defined in: `demos/demo_utils/yolo_cfg.cpp`  
- Parameters referenced in: `yolo_pose.cpp`   

```
// pre/post parameter table
YoloParam yoloParams[] = {
    [0] = yolov5s6_pose_640, // ----> p option : 0
};
```

To configure custom pose estimation parameters  

- Modify the existing entries in the configuration files, or  
- Add new model-specific configurations  

This setup ensures that the NPU output is properly decoded into keypoint coordinates and bounding boxes.  


### Segmentation  
This section explains how to run semantic segmentation demos based on YOLOv5 models. Both semantic segmentation models are supported.  

- Model: `DeepLabV3PlusMobileNetV2_2.dxnn`  

**Semantic Segmentation - Cityscape Dataset-Based Demo**  
This demo describes an example of semantic segmentation based on the DeepLabV3Plus model trained on the **Cityscape** dataset, designed for detailed urban scene parsing.  

- Model: `DeepLabV3PlusMobileNetV2_2.dxnn`  
- Path: `assets/models`  
- Dataset: Trained on the **Cityscape** dataset  

This model performs pixel-wise classification, assigning a semantic label to each pixel in the input image. This allows for dense and structured scene understanding.  

**Notes.**  

- Load `DeepLabV3PlusMobileNetV2_2.dxnn` in the segmentation pipeline.  
- Use the **Cityscape** class index mappings to interpret the output mask.  

```
/* class_index, class_name, colorB, G, R */
SegmentationParam segCfg[] = {
    { 0 , "road" , 128 , 64 , 128 , },
    { 1 , "sidewalk" , 244 , 35 , 232 , },
    { 2 , "building" , 70 , 70 , 70 , },
    { 3 , "wall" , 102 , 102 , 156 , },
    { 4 , "fence" , 190 , 153 , 153 , },
    { 5 , "pole" , 153 , 153 , 153 , },
    { 6 , "traffic light" , 51 , 255 , 255 , },
    { 7 , "traffic sign" , 220 , 220 , 0 , },
    { 8 , "vegetation" , 107 , 142 , 35 , },
    { 9 , "terrain" , 152 , 251 , 152 , },
    { 10 , "sky" , 255 , 0 , 0 , },
    { 11 , "person" , 0 , 51 , 255 , },
    { 12 , "rider" , 255 , 0 , 0 , },
    { 13 , "car" , 255 , 51 , 0 , },
    { 14 , "truck" , 255 , 51 , 0 , },
    { 15 , "bus" , 255 , 51 , 0 , },
    { 16 , "train" , 0 , 80 , 100 , },
    { 17 , "motorcycle" , 0 , 0 , 230 , },
    { 18 , "bicycle" , 119 , 11 , 32 , },
};
```

```
./bin/segmentation -m assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -i sample/8.jpg
```

![](./../resources/03_05_Ouput_of_Segmentation.png){ width=600px }


**Semantic Segmentation - Object Detection Demo**  
This demo explains segmentation capabilities by combining semantic segmentation and object detection.  

- Segmentation Model: `DeepLabV3PlusMobileNetV2_2.dxnn`  
- Object Detection Model: YOLOv5  

How It Works,  

- The segmentation model classifies every pixel into semantic categories such as road, person, or building  
- The detection model locates and classifies objects such as e.g., cars, traffic signs  

Benefits of Combined Pipeline  

- Semantic segmentation provides context-aware, fine-grained understanding of background and scene layout  
- Object detection focuses on identifying and locating distinct object instances  

```
./bin/od_segmentation -m0 assets/models/YOLOV5S_3.dxnn -p0 1 -m1 assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -i sample/8.jpg
```

![](./../resources/03_06_Output_of_od_segmentation.png){ width=600px }

---
