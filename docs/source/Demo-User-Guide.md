This is an application examples that gives you a quick experience of NPU Accelerator performance. 
You can refer to **DX-APP** and modify it a little or implement application depending on the purpose of your use. 
This can reduce stress, such as setting the environment and implementing the code. 
Application performance may also depending on the specifications of the host CPU because it includes pre/post processing and graphics processing operations. 

## C++ Demo Application List

**Classification Demo**

- Basic Classification

- ImageNet Classification

**Object Detection Demo**

- Yolo Object Detection

- Yolo Object Detection - Multi Channel

**Pose Estimation Demo**

- Human Pose Estimation

**Segmentation Demo**

- Semantic Segmentation PIDNet (CityScape dataset Only)

- Semantic Segmentation PIDNet (CityScape dataset Only) + Yolo Object Detection

## Run the Demo Executable
**Getting the usage of executable, Try run with "-h" option.**

### Run Classification

**Classification**

```shell 
./bin/classification -m assets/models/EfficientNetB0_4.dxnn -i example/ILSVRC2012/1.jpeg
```
in Windows
```shell
bin\classification.exe -m assets/models/EfficientNetB0_4.dxnn -i example/ILSVRC2012/1.jpeg
```
The following result can be displayed : 
```shell
  Top1 Result : class 321
```

**ImageNet Classification** 

```shell
./bin/imagenet_classification -m assets/models/EfficientNetB0_4.dxnn -i example/imagenet_val_map.txt -p example/ILSVRC2012/
```
![](./resources/result_imagenet.jpg)

### Run Object Detection

**Yolo Object Detection**

Example excutable for yolov5s_512 model.

```shell 
./bin/yolo -m assets/models/YOLOV5S_3.dxnn -i sample/1.jpg -p 1
...
  Detected 8 boxes.
    BBOX:person(0) 0.859366, (307.501, 138.443, 400.977, 364.696)
    BBOX:oven(69) 0.661055, (-0.446404, 225.652, 155.377, 325.085)
    BBOX:bowl(45) 0.564862, (46.2462, 314.978, 105.182, 347.728)
    BBOX:person(0) 0.561198, (0.643028, 295.378, 47.8478, 331.855)
    BBOX:oven(69) 0.494507, (390.414, 245.532, 495.489, 359.54)
    BBOX:bowl(45) 0.47086, (-0.300865, 328.801, 69.139, 379.72)
    BBOX:bowl(45) 0.452027, (25.9788, 359.192, 80.7059, 392.734)
    BBOX:pottedplant(58) 0.368497, (0.423544, 86.835, 51.0048, 206.592)
```

![](./resources/result_yolo.jpg)


Yolo Object Detection applications requires pre/post processing parameters that are not included in the compiled model.

Example paramters are describes in yolo_cfg.cpp and are all listed in yolo_demo.cpp as follows :

```cpp
// pre/post parameter table
YoloParam yoloParams[] = {
    [0] = yolov5s_320,
    [1] = yolov5s_512,             // ----> p option : 1
    [2] = yolov5s_640,
    [3] = yolox_s_512,
    [4] = yolov7_640,
    [5] = yolov7_512,
    [6] = yolov4_608,
};
```

To configure your own parameters, simply modify the examples or add new examples to the list and yolo_cfg.cpp.

This is Example config for yolov5s_512 model.

```cpp
YoloParam yolov5s_512 = {
    .height = 512,
    .width = 512,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1,
    .numClasses = 80,
    .layers = {
        {
            .numGridX = 64,
            .numGridY = 64,
            .numBoxes = 3,
            .anchorWidth = { 10.0, 16.0, 33.0 },
            .anchorHeight = { 13.0, 30.0, 23.0 },
            .tensorIdx = { 0 },
        },
        .
        .
        .
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle", . . .}
};
```

**Yolo Multi Channel Object Detection**

Example excutable for yolov5s_512 model.

```shell
./bin/yolo_multi -c ./example/yolo_multi_demo.json
```

![](./resources/result_yolo_multi.jpg)

As like above yolo model list, You need to modify or create a json config file. 
The example json file is located in *example/yolo_multi_input_source_demo.json* and must be modified for use. 

Refer to following configuration json file. 

When in `offline` mode and a video file is inserted in `video_sources`, you can specify the number of frames in the third parameter. 
This allows for pre-processing and inference on the specified number of frames in the video. 

```json
{
    "usage": "multi",
      "model_path": "/model_path/yolov5s_512",
      "model_name": "yolov5s_512",
      
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

Set "model_name" for mapping pre/post processing parameters.

```cpp
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

You can run a demo using an `RTSP` video stream by specifying the `RTSP URL` and the network type (e.g., "rtsp").

Use the following json file : 

```json
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

### Run Pose Estimation 

**Pose Estimation** 

This project was produced with reference to Ultralytics yolo pose model. 
Example excutable for yolo pose model.

```shell
./bin/pose -m assets/models/YOLOV5Pose640_1.dxnn -i sample/1.jpg -p 0
```

![](./resources/result_yolo_pose.jpg)


The dxrt model has input and output tensors that shapes are N H W C format by default. 

Due to the characteristics of the NPU, if the channel size is less than 64 bytes, it is aligned to 16 bytes, and if the channel size is 64 bytes or larger, it is aligned to 64 bytes.

For example, **[1, 40, 52, 36]** will be **[1, 52, 36, 40 + 24]** output tensor which is with 24 bytes of dummy.

```text
  inputs
    images, INT8, [1, 640, 640, 3,  ], 0
  outputs
    /0/model.33/m_kpt.0/Conv_output_0, FLOAT, [1, 80, 80, 192,  ], 0 ----> Key points per 3 anchor boxes
    /0/model.33/m.0/Conv_output_0, FLOAT, [1, 80, 80, 64,  ], 0      ----> Object Box Info 
    /0/model.33/m_kpt.1/Conv_output_0, FLOAT, [1, 40, 40, 192,  ], 0
    /0/model.33/m.1/Conv_output_0, FLOAT, [1, 40, 40, 64,  ], 0
    /0/model.33/m_kpt.2/Conv_output_0, FLOAT, [1, 20, 20, 192,  ], 0
    /0/model.33/m.2/Conv_output_0, FLOAT, [1, 20, 20, 64,  ], 0
    /0/model.33/m_kpt.3/Conv_output_0, FLOAT, [1, 10, 10, 192,  ], 0
    /0/model.33/m.3/Conv_output_0, FLOAT, [1, 10, 10, 64,  ], 0
```

Yolo Pose applications requires pre/post processing parameters that are not included in the compiled model. 

Example paramters are describes in pose_estimation/yolo_cfg.cpp and are all listed in pose_demo.cpp as follows :

```cpp
// pre/post parameter table
YoloParam yoloParams[] = {
    [0] = yolov5s6_pose_640,         // ----> p option : 0
    [1] = yolov5s6_pose_1280
};
```

### Run Segmentation

**Semantic Segmentation cityscape dataset based model** 

This project was produced with reference to DeepLabV3Plus model.

Example excutable for semantic segmentation model. 

You can use the DeepLabV3PlusMobileNetV2_2.dxnn model included with assets/models. this model trained with the **cityscape dataset**, a segmentation config may be generated by referring to below.

```cpp
/* class_index, class_name, colorB, G, R */
SegmentationParam segCfg[] = {
    {	0	  ,	"road"          ,	128	,	64	,	128	,	},
    {	1	  ,	"sidewalk"      ,	244	,	35	,	232	,	},
    {	2	  ,	"building"      ,	70	,	70	,	70	,	},
    {	3	  ,	"wall"          ,	102	,	102	,	156	,	},
    {	4	  ,	"fence"         ,	190	,	153	,	153	,	},
    {	5	  ,	"pole"          ,	153	,	153	,	153	,	},
    {	6	  ,	"traffic light" ,	51	,	255	,	255	,	},
    {	7	  ,	"traffic sign"  ,	220	,	220	,	0	,	},
    {	8	  ,	"vegetation"    ,	107	,	142	,	35	,	},
    {	9	  ,	"terrain"       ,	152	,	251	,	152	,	},
    {	10	,	"sky"           ,	255	,	0	  ,	0	,	},
    {	11	,	"person"        ,	0	  ,	51	,	255	,	},
    {	12	,	"rider"         ,	255	,	0	  ,	0	,	},
    {	13	,	"car"           ,	255	,	51	,	0	,	},
    {	14	,	"truck"         ,	255	,	51	,	0	,	},
    {	15	,	"bus"           ,	255	,	51	,	0	,	},
    {	16	,	"train"         ,	0	  ,	80	,	100	,	},
    {	17	,	"motorcycle"    ,	0	  ,	0	  ,	230	,	},
    {	18	,	"bicycle"       ,	119	,	11	,	32	,	},
};
```

```shell
./bin/pidnet -m assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -i sample/8.jpg
```

![](./resources/result_cityscape_seg.jpg)

**Semantic Segmentation with Object Detection** 

This project was produced with reference to DDRNet and yolov5 models.

```shell
./bin/od_pid -m0 assets/models/YOLOV5S_3.dxnn -m1 assets/models/DeepLabV3PlusMobileNetV2_2.dxnn -i sample/8.jpg
``` 

![](./resources/result_yolo_cityscape_seg.jpg)

### Run Face Recognition

**Face Recognition**

This uses a total of three models and consists of a face detection, a face landmark, and a face vector extraction model. 
This demo is a face recognition project, and you can also experience demo with the existing Face database or without database. 
If you want to use Face DB, you can enter the Data Base path with the *'-p'* option but it is not a required option. 
A description of the parameter can be found by entering the option *'-h'*. 
The face detection algorithm used a model of SSD network construction. Customization is essential for the user. 
Face Recognition demo has two functions in total.

**Measure face similarity by two images**

```shell
./bin/face_recognition -m0 /your-face-detection-model-path/graph.dxnn -m1 /your-face-align-model-path/graph.dxnn -m2 /your-face-vector-model-path/graph.dxnn -l /image1-to-compare.jpg -r /image2-to-compare.jpg 
```

**Tracking similar faces from camera by searching face database**

```shell
./bin/face_recognition -m0 /your-face-detection-model-path/graph.dxnn -m1 /your-face-align-model-path/graph.dxnn -m2 /your-face-vector-model-path/graph.dxnn -c -t -p /your-face-database-path/ 
```

![](./resources/result_faceID.jpg)

