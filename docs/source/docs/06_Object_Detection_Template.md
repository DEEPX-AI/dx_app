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
- `yolo_face`: For face detection models  

Post-processing parameters, such as thresholds and class names, can be adjusted in the JSON file without recompilation.

---

## Custom Post-Processing Guide for Your Models

This guide provides instructions for customizing the post-processing pipeline to suit your model architecture and deployment configuration.  
Post-processing behavior may vary depending on the model’s output shape and the DeepX NPU execution environment. It is essential to understand your model’s structure and configure post-processing accordingly.  

Supported scenarios include  

- Post-Processing with `USE_ORT=ON`  
- Post-Processing with `USE_ORT=OFF`  
- Post-Processing with `PPU` (Post Processing Unit instead CPU)
- (Optional) Custom Post-Processing  

---

## Post-Processing Using USE_ORT=ON

If the DXRT framework is executed with `USE_ORT=ON`, the output will be in the same format as the original ONNX model. Therefore, you can directly apply your existing ONNX post-processing pipeline.

And Applying NMS  
After struct conversion, the `cxcywh` format **must** be translated to corner format (`xmin, ymin, xmax, ymax`). This enables accurate rendering and overlap calculations.  
Following decoding, apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes and retain only high-confidence predictions.  

Refer to `dx_app/lib/post_process/yolo_post_processing.hpp` - line.204 for the Reference Implementation.  

---

## Post-Processing Using USE_ORT=OFF

If executed with `USE_ORT=OFF`, the model output will consist of three blobs.
```
inputs
    images, UINT8, [1, 512, 512, 3, ], 0

outputs
    378, FLOAT, [1, 255, 64, 64, ], 0
    439, FLOAT, [1, 255, 32, 32, ], 0
    500, FLOAT, [1, 255, 16, 16, ], 0
```

Required Post-Processing Steps  

- **1.** Anchor Box Scaling and Offset Recovery: Reconstruct the predicted bounding box coordinates using the anchor box configuration and grid position.  
- **2.** Sigmoid Activation: Apply the sigmoid function to bounding box offsets , objectness scores, and  class probabilities.  
- **3.** Non-Maximum Suppression (NMS): Filter overlapping boxes and retain only the highest-confidence detections.  

Refer to `dx_app/lib/post_process/yolo_post_processing.hpp` for implementation details.  

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

An example model using PPU is `YOLOV5S_PPU.dxnn`. Running the example command is as follows.  
```
run_model -m ./assets/models/YOLOV5S_PPU.dxnn
```

The output of the example model is as follows.  
```
  Inputs
     -  images, UINT8, [1, 320, 960 ]
  Outputs
    -  BBOX, BBOX, [1, unknown ]
```

Accessing PPU Output  
The PPU format is available in `datatypes.h` dxrt include path. You can access the result data using the following code.  
```cpp
#include <dxrt/dxrt_api.h>

auto outputs = ie.Run(input_data);
dxrt::DeviceBoundingBox_t* raw_data =
  static_cast<dxrt::DeviceBoundingBox_t*>(outputs.front()->data());
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
The inference result from PPU is **not** returned in a standard multi-dimensional tensor format (e.g., `[N, C, H, W]`). Instead, the output is shaped as `[1, num_boxes]`.  

Each entry represents a bounding box and contains the following fields  

- `x, y, w, h`: Bounding box in center format (`cxcywh`)  
- `grid`: Grid cell index  
- `box_idx`: Anchor index  
- `layer_idx`: Detection layer index  
- `score`: Confidence score  
- `label`: Class ID  
- `padding[4]`: 4-byte alignment for 32-byte struct size  

You will need to use a loop to convert the output into the `DeviceBoundingBox_t` structure and then handle post-processing from there.  

```cpp
auto outputs = ie.run(input_data);
dxrt::DeviceBoundingBox_t* raw_data = static_cast<dxrt::DeviceBoundingBox_t*>(outputs.front()->data());
for (int i = 0; i < outputs.front()->shape()[0]; i++) {
    auto data = raw_data[i];
    // Your post-processing logic here
}
```

Refer to `dx_app/demos/demo_utils/yolo.cpp` - line.277 for the following example.  
```cpp
/* Example of dxrt::DeviceBoundingBox_t */
box_temp[0] = (data[i].x * 2. - 0.5 + gX) * stride; // cx
box_temp[1] = (data[i].y * 2. - 0.5 + gY) * stride; // cy
box_temp[2] = pow((data[i].w * 2.), 2) * layer.anchorWidth[data[i].box_idx]; // w
box_temp[3] = pow((data[i].h * 2.), 2) * layer.anchorHeight[data[i].box_idx]; // h

x1 = box_temp[0] - box_temp[2] / 2.; /*x1*/
y1 = box_temp[1] - box_temp[3] / 2.; /*y1*/
x2 = box_temp[0] + box_temp[2] / 2.; /*x2*/
y2 = box_temp[1] + box_temp[3] / 2.; /*y2*/
```

Decoding and Applying NMS  
After struct conversion, the `cxcywh` format **must** be translated to corner format (`xmin, ymin, xmax, ymax`). This enables accurate rendering and overlap calculations.  
Following decoding, apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes and retain only high-confidence predictions.  

Refer to `dx_app/demos/demo_utils/yolo.cpp` - line.154 for the Reference Implementation.  

---

## (Optional) Custom Post-Processing

If you are working with a custom detection model that does not follow the standard YOLO output structure, you can define your own post-processing logic.  

**Step 1. Set custom_decode in JSON**  
In your configuration file (e.g., `yolov5s3_example.json`), set the decoding_method field to `decoding_method : custom_decode`. This tells the SDK to bypass default decoding and call your custom implementation.  

**Step 2. Modify getBoxesFromCustomPostProcessing()**  
Update the following function to handle your model’s output format  

- File: `dx_app/lib/post_process/yolo_post_processing.hpp` - line.630

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
dxapp::common::BBox yoloCustomDecode(std::function<float(float)> activation, std::vector<float> datas, dxapp::common::Point grid, dxapp::common::Size
anchor, int stride, float scale)
{
    /**
      * @brief adding your decode method
      *
      * example code ..
          * dxapp::common::BBox box_temp;
          * box_temp._xmin = (activation(datas[0]) * 2. - 0.5 + grid._x ) * stride; //center x
          * box_temp._ymin = (activation(datas[1]) * 2. - 0.5 + grid._y ) * stride; //center y
          * box_temp._width = std::pow((activation(datas[2]) * 2.f), 2) * anchor._width;
          * box_temp._height = std::pow((activation(datas[3]) * 2.f), 2) * anchor._height;
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
