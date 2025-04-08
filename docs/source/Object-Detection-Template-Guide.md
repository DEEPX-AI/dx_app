## Run Object Detection Template (Yolo Series)

```shell
./bin/run_detector -c example/yolov5s3_example.json
  ...
  detected : 9
```

*example/yolov5s3_example.json* is a json config file that can run the detection model. Referring to this, you can customizing input and output. 

If "type" is "save," the output is saved as a video file. If "type" is "realtime," the output is displayed on the screen. If "type" is "none," the count of detection results is continuously printed. 
The current example json file has the "type" set to "none." If you want to save the output, change it to "save." 

When using the yolo model, the post processing parameters can be modified to suit the use by referring to official yolov5 model. 
And You can also modify classes information. 
You don't have to revise only the json config file to recompile the code. 
The "yolo_basic" method uses the decoding approach from YOLOv3, YOLOv5, YOLOv7. Other methods include "yolo_scale," "yolox," "yolo_pose," "yolov8," and "scrfd."

## Custom Post-Processing Guide for Your Models 

This guide provides instructions on how to port post-processing for your model. 
It's essential to fully understand the model, as the output shape may differ depending on the NPU configuration 
and the following three options:   

- **Post-Processing Using PPU**
- **Post-Processing with `USE_ORT=ON`**
- **`USE_ORT=OFF` and Without PPU**
- **(option) Custom Post-Processing**

## Post-Processing Using PPU   
The Post-Processing Unit (PPU) calculates scores, threshold filtering, and extracts only the valid bounding box(BBox). 

The output from the PPU will be as follows:

- Filtered boxes in `xywh` format (depending on whether the format is `cxcywh`, `x1y1x2y2`, etc)

- BBoxes' scores 

- BBoxes' locations

You only need to perform BBox decoding and Non-Maximum Suppression (NMS) on the filtered boxes and their class indices. 
This process minimizes memory usage via PCIe, reduces latency, and requires only a single loop during post-processing. 

To utilize PPU during model compilation, ensure the model meets the PPU usage conditions, 
and configure the `pp` field in `config.json` as shown below: 

```json
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

An example model using PPU is `YOLOV5S_3.dxnn`. Running the example command:

```shell
run_model -m ./assets/models/YOLOV5S_3.dxnn
```

It will output: 

```shell
inputs
  images, INT8, [1, 512, 512, 3,  ], 0
outputs
  DX_tensor_3748, BBOX, [-1,  ], 0
```

The PPU format is available in *dx_rt/lib/include/dxrt/datatypes.h*. 
You can access the result data using the following code:

```cpp
#include "dxrt/dxrt_api.h"

auto outputs = ie.run(input_data);
dxrt::DeviceBoundingBox_x* raw_data = 
    static_cast<dxrt::DeviceBoundingBox_x*>(outputs.front()->data());
```

The structure used for bounding boxes is: 
```cpp
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

The post-processing uses values for x, y, w, h, grid, box_idx (anchor index), layer_idx (layer index), score, and label. 

To keep everything aligned to 32 bytes, an extra 4 bytes of padding are added.

The inference output is not in the usual shape like (x, x, x, x). Instead, it's shaped as (number of boxes, 1). 

You will need to use a loop to convert the output into the DeviceBoundingBox_t structure and then handle post-processing from there.

```cpp
auto outputs = ie.run(input_data);
dxrt::DeviceBoundingBox_x* raw_data = static_cast<dxrt::DeviceBoundingBox_x*>(outputs.front()->data());
for (int i = 0; i < outputs.front()->shape()[0]; i++ ){
    dxrt::DeviceBoundingBox_x* data = raw_data + i
    ....
}
```

Refer to *dx_app/lib/post_process/yolo_post_processing.hpp*, l.246 for the following example:

```cpp
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

In this code, the cxcy format data (center x and y coordinates, width, and height) is converted into bounding box coordinates: 

xmin, ymin, xmax, and ymax, along with width and height.
   
After this, Non-Maximum Suppression (NMS) is applied to remove overlapping boxes and keep the best ones.

## Post-Processing with `USE_ORT=ON`
If the **PPU** option is not used, and the model is executed with `USE_ORT=ON`, 
the output will be similar to the original ONNX model, and you can apply your existing post-procesing pipeline.

## `USE_ORT=OFF` and Without PPU

If neither option is used, the model output will consist of three blobs:

```shell
inputs
  images, UINT8, [1, 512, 512, 3,  ], 0
outputs
  378, FLOAT, [1, 64, 64, 256,  ], 0
  439, FLOAT, [1, 32, 32, 256,  ], 0
  500, FLOAT, [1, 16, 16, 256,  ], 0
```

Due to the NPU alignment constraints, the output channels will be 256 instead of the typical 255. 

Post-processing includes the following steps:

- **Anchor Box Scaling and Offset Calculation**: Restore predicted bounding box coordinates.

- **Sigmoid Activation**: Apply sigmoid to bounding box coordinates and class probabilities.

- **Non-Maximum Suppression (NMS)**: Retain only the most confident predictions for overlapping boxes.

Refer to *dx_app/lib/post_process/yolo_post_processing.hpp* for implementation details.

## Custom Post-Processing 

If you're working with a custom model, rather than a standard YOLO series, you will need to modify the post-processing template.

Start by changing the `decoding_method` to `custom_decode` in the yolov5s3_example.json file. 

Then, update the function `getBoxesFromCustomPostProcessing` found in **dx_app/lib/post_process/yolo_post_processing.hpp**, line 653. 

You can add any necessary parameters here.

```cpp
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

You can also customize the `yoloCustomDecode` function in *dx_app/lib/utils/box_decode.hpp* as needed.

Write your own decode function by analyzing the following code block.

The code block below is located in path *lib/utils/box_decode.hpp*.

```cpp
dxapp::common::BBox yoloCustomDecode(std::function<float(float)> activation, std::vector<float*> datas, dxapp::common::Point grid, dxapp::common::Size anchor, int stride, float scale)
{
    /**
      * @brief adding your decode method
      *
      * example code ..
      *      float* data = datas[0];
      *      dxapp::common::BBox box_temp;
      *      box_temp._xmin = (activation(data[0]) * 2. - 0.5 + grid._x ) * stride; //center x
      *      box_temp._ymin = (activation(data[1]) * 2. - 0.5 + grid._y ) * stride; //center y
      *      box_temp._width = std::pow((activation(data[2]) * 2.f), 2) * anchor._width;
      *      box_temp._height = std::pow((activation(data[3]) * 2.f), 2) * anchor._height;
      *      dxapp::common::BBox result = {
      *              ._xmin=box_temp._xmin - box_temp._width / 2.f,
      *              ._ymin=box_temp._ymin - box_temp._height / 2.f,
      *              ._xmax=box_temp._xmin + box_temp._width / 2.f,
      *              ._ymax=box_temp._ymin + box_temp._height / 2.f,
      *              ._width = box_temp._width,
      *              ._height = box_temp._height,
      *      };
      *
      */

    dxapp::common::BBox result;

    return result;
};
```

As you know, if you modify the code, you have to **re-compile** it.

```shell
./build.sh
```