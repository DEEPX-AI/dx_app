This section explains how to use the Object Detection Template to execute YOLO-series models on DeepX NPUs by configuring only a JSON file, thus no source code modifications are needed.

---

## Run Object Detection Template (Yolo Model)

To run an object detection model using YOLOv5.

```
./bin/run_detector -c example/run_detector/yolov5s3_example.json
  ...
  detected : 9
```

Configuration File  

- Path: `example/run_detector/yolov5s3_example.json`  
- This file defines  all runtime parameters including model path, input/output shape, post-processing method, and output type  

Output Display Options  

- `type value`:	Description  
- `save`:	Saves output to a video file  
- `realtime`:	Displays results in a window (GUI)  
- `none`:	Continuously prints detection count (default)  

**Note.** The current example file sets `type: none`. To save video output, change it to `save`.  

Supported Post-Processing Methods  
The detection template supports various decoding methods tailored to different YOLO variants.  

- `yolo_basic`: For YOLOv3, YOLOv5, YOLOv7  
- `yolo_scale`: Multi-scale YOLO  
- `yolox`: For YOLOX  
- `yolo_pose`: For human pose estimation  
- `yolov8`: For YOLOv8 models  
- `scrfd`: For face detection models  

Post-processing parameters, such as thresholds and class names, can be adjusted in the JSON file without recompilation.

---

## Custom Post-Processing Guide for Your Models

This guide provides instructions for customizing the post-processing pipeline to suit your model architecture and deployment configuration.  
Post-processing behavior may vary depending on the model’s output shape and the DeepX NPU execution environment. It is essential to understand your model’s structure and configure post-processing accordingly.  

Supported scenarios include  

- Post-Processing Using PPU  
- Post-Processing with `USE_ORT=ON`  
- Post-Processing with `USE_ORT=OFF` and No PPU  
- (Optional) Custom Post-Processing  

---

## Post-Processing Using PPU

The Post-Processing Unit (PPU) is a hardware-accelerated module that performs the confidence score calculation, the threshold filtering, and the extraction of valid bounding boxes (BBox).  

PPU Output Format  
The PPU outputs a list of filtered bounding boxes containing  

- Box coordinates (in `xywh`, `cxcywh`, or `x1y1x2y2` format)  
- Class labels  
- Confidence scores  

Developer Action  
After receiving the PPU output, only the following steps are required  

- **1.** BBox decoding (format conversion: e.g., `cxcywh` → `xmin`, `ymin`, `xmax`, `ymax`)  
- **2.** Non-Maximum Suppression (NMS)  

This reduces PCIe bandwidth usage and minimizes latency by limiting data transferred to the host.  
A single loop can complete post-processing.  

Enabling PPU in `config.json`  
To use PPU during model compilation, ensure the model structure meets PPU constraints, and include the pp field in your config.  
`config.json` as shown below.  

```
{
  ...
  "pp": {
    "conf_thres": 0.25,
    "iou_thres": 0.4,
    "max_det": 1000,
    "activation": "Sigmoid",
    "decoding_type": "yolo_a",
    "box_format": "center",
    "layer": {
      "Conv_343": {
        "anchor_width": [116, 156, 373],
        "anchor_height": [90, 198, 326],
        "stride": 32
      },
      "Conv_294": {
        "anchor_width": [30, 62, 123],
        "anchor_height": [40, 80, 160],
        "stride": 16
      }
    }
  }
}
```

An example model using PPU is `YOLOV5S_3.dxnn`. Running the example command is as follows.  
```
run_model -m ./assets/models/YOLOV5S_3.dxnn
```

The output of the example model is as follows.  
```
inputs
  images, INT8, [1, 512, 512, 3, ], 0
outputs
  DX_tensor_3748, BBOX, [-1, ], 0
```

Accessing PPU Output  
The PPU format is available in `dx_rt/lib/include/dxrt/datatypes.h`. You can access the result data using the following code.  
```
#include "dxrt/dxrt_api.h"

auto outputs = ie.run(input_data);
dxrt::DeviceBoundingBox_x* raw_data =
  static_cast<dxrt::DeviceBoundingBox_x*>(outputs.front()->data());
```

The structure used for bounding boxes is as follows.  
```
typedef struct DXRT_API {
    float x;
    float y;
    float w;
    float h;
    uint8_t grid_y;
    uint8_t grid_x;
    uint8_t box_idx;
    uint8_t layer_idx;
    float score;
    uint32_t label;
    char padding[4];
} DeviceBoundingBox_t;
```

Interpreting PPU Output  
The inference result from PPU is **not** returned in a standard multi-dimensional tensor format (e.g., `[N, C, H, W]`). Instead, the output is shaped as `[num_boxes, 1]`.  

Each entry represents a bounding box and contains the following fields  

- `x, y, w, h`: Bounding box in center format (`cxcywh`)  
- `grid`: Grid cell index  
- `box_idx`: Anchor index  
- `layer_idx`: Detection layer index  
- `score`: Confidence score  
- `label`: Class ID  
- `padding[4]`: 4-byte alignment for 32-byte struct size  

You will need to use a loop to convert the output into the `DeviceBoundingBox_t` structure and then handle post-processing from there.  

```
auto outputs = ie.run(input_data);
dxrt::DeviceBoundingBox_x* raw_data = static_cast<dxrt::DeviceBoundingBox_x*>(outputs.front()->data());
for (int i = 0; i < outputs.front()->shape()[0]; i++ ){
    dxrt::DeviceBoundingBox_x* data = raw_data + i
    ....
}
```

Refer to `dx_app/lib/post_process/yolo_post_processing.hpp` - line.246 for the following example.  

```
dxapp::common::BBox temp;
temp = {
    (data->x * 2 - 0.5f + numGridX) * stride,
    (data->y * 2 - 0.5f + numGridY) * stride,
    0,
    0,
    (data->w * data->w * 4) * layer.anchor_width[data->box_idx],
    (data->h * data->h * 4) * layer.anchor_height[data->box_idx],
    {dxapp::common::Point_f(-1, -1, -1)}
};
temp._xmin = temp._xmin - (temp._width / 2);
temp._ymin = temp._ymin - (temp._height / 2);
temp._xmax = temp._xmin + (temp._width / 2);
temp._ymax = temp._ymin + (temp._height / 2);
```

Decoding and Applying NMS  
After struct conversion, the `cxcywh` format **must** be translated to corner format (`xmin, ymin, xmax, ymax`). This enables accurate rendering and overlap calculations.  
Following decoding, apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes and retain only high-confidence predictions.  

Refer to `dx_app/lib/post_process/yolo_post_processing.hpp` - line.246 for the Reference Implementation.  

---

## Post-Processing Using USE_ORT=ON

If the PPU option is **not** used, and the model is executed with `USE_ORT=ON`, the output will be similar to the original ONNX model, and you can apply your existing post-procesing pipeline.

---

## Post-Processing Using USE_ORT=OFF and No PPU

If PPU is **disabled** and the model is executed with `USE_ORT=OFF`, the model output will consist of three blobs.
```
inputs
    images, UINT8, [1, 512, 512, 3, ], 0

outputs
    378, FLOAT, [1, 64, 64, 256, ], 0
    439, FLOAT, [1, 32, 32, 256, ], 0
    500, FLOAT, [1, 16, 16, 256, ], 0
```

Due to NPU alignment constraints, output channel dimensions may be padded.  

- For example, a YOLO model that normally outputs 255 channels will produce 256 channels instead.  
- The extra channel **should be ignored** during post-processing.  

Required Post-Processing Steps  

- **1.** Anchor Box Scaling and Offset Recovery: Reconstruct the predicted bounding box coordinates using the anchor box configuration and grid position.  
- **2.** Sigmoid Activation: Apply the sigmoid function to bounding box offsets , objectness scores, and  class probabilities.  
- **3.** Non-Maximum Suppression (NMS): Filter overlapping boxes and retain only the highest-confidence detections.  

Refer to `dx_app/lib/post_process/yolo_post_processing.hpp` for implementation details.  

---

## (Optional) Custom Post-Processing

If you are working with a custom detection model that does not follow the standard YOLO output structure, you can define your own post-processing logic.  

**Step 1. Set custom_decode in JSON**  
In your configuration file (e.g., `yolov5s3_example.json`), set the decoding_method field to `decoding_method : custom_decode`. This tells the SDK to bypass default decoding and call your custom implementation.  

**Step 2. Modify getBoxesFromCustomPostProcessing()**  
Update the following function to handle your model’s output format  

- File: `dx_app/lib/post_process/yolo_post_processing.hpp` - line.653

You can add any necessary parameters here.  

```
void getBoxesFromCustomPostProcessing(uint8_t* outputs /* Users can add necessary parameters manually. */)
{
    /**
        * @brief adding your post processing code
        *
        * example code ..
        *
        * int boxIdx = 0;
        * float* node_a = (float*)outputs;
        * float* node_b = (float*)node_a + node_a_size;
        * float* node_c = (float*)node_b + node_b_size;
        * for(int i=0; i<node_length; i++)
        * {
        *
        * }
        *
        *
        */
};
```

**Step 3.  (Optional) Modify yoloCustomDecode()**  
For more advanced control, modify the low-level decoding logic in `dx_app/lib/utils/box_decode.hpp`. You can use this function to implement custom decoding logic beyond the default YOLO box structure.  

```
dxapp::common::BBox yoloCustomDecode(std::function<float(float)> activation, std::vector<float*> datas, dxapp::common::Point grid, dxapp::common::Size
anchor, int stride, float scale)
{
    /**
      * @brief adding your decode method
      *
      * example code ..
          * float* data = datas[0];
          * dxapp::common::BBox box_temp;
          * box_temp._xmin = (activation(data[0]) * 2. - 0.5 + grid._x ) * stride; //center x
          * box_temp._ymin = (activation(data[1]) * 2. - 0.5 + grid._y ) * stride; //center y
          * box_temp._width = std::pow((activation(data[2]) * 2.f), 2) * anchor._width;
          * box_temp._height = std::pow((activation(data[3]) * 2.f), 2) * anchor._height;
          * dxapp::common::BBox result = {
              * ._xmin=box_temp._xmin - box_temp._width / 2.f,
              * ._ymin=box_temp._ymin - box_temp._height / 2.f,
              * ._xmax=box_temp._xmin + box_temp._width / 2.f,
              * ._ymax=box_temp._ymin + box_temp._height / 2.f,
              * ._width = box_temp._width,
              * ._height = box_temp._height,
      * };
      *
      */

    dxapp::common::BBox result;

    return result;

};
```

**Note.** Once you modify the code, you **must** recompile it.
```
./build.sh
```

---
