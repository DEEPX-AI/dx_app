This chapter focuses on accelerating YOLO post-processing in Python using Pybind11. It introduces the `dx_postprocess` module and the `YoloPostProcess` class, lists supported models, and explains synchronous/asynchronous usage with examples.

---

## YOLO Post-Processing Optimization Using Pybind11

### Overview

Post-processing for YOLO models can be implemented in Python using libraries such as numpy and `torchvision.ops.nms`. This is demonstrated in [`templates/python/yolov5s_example.py`](../../../templates/python/yolov5s_example.py) and [`/templates/python/yolov_async.py`](../../../templates/python/yolo_async.py).  

However, these Python-based implementations may not offer optimal performance in terms of latency, particularly for real-time applications.  

To maximize FPS (Frames Per Second) in time-sensitive scenarios, it's recommended to implement performance-critical post-processing steps—such as score filtering, bounding box decoding, and Non-Maximum Suppression (NMS)—in C++, and integrate them into Python using Pybind11.  

This approach significantly reduces latency by offloading heavy computations to native C++ code while maintaining the usability of Python.  


The typical YOLO post-processing pipeline consists of the following steps.  

- Score filtering  
- Bounding box decoding  
- NMS (Non-Maximum Suppression)  


These steps involve intensive matrix operations and iterative loops, making them performance-critical. Executing them in C++ is significantly more efficient than in Python, especially for real-time applications.  

To address this, DX-APP provides the **`dx_postprocess`** module, which includes the **`YoloPostProcess`** class, an optimized C++ implementation of YOLO post-processing exposed to Python with Pybind11.  

By using **`YoloPostProcess`**, users can perform end-to-end YOLO post-processing with a single function call, supporting a variety of YOLO model types with minimal effort and maximum performance.  


###  Supported Use Cases for  YoloPostProcess

**`YoloPostProcess`** demonstrates how C++ code can be wrapped with Pybind11 to optimize post-processing in Python environments. It shows how to implement performance-critical post-processing steps (score filtering, bounding box decoding, NMS, etc.) in C++ and make them available for use in Python.

**Important**: **`YoloPostProcess`** is not a universal solution that supports post-processing for all YOLO series models. It is an example implementation designed to work only with specific models and configurations.

Among the models provided by DX-APP, the following cases support **`YoloPostProcess`**:

| Model                    | USE_ORT = ON | USE_ORT = OFF | Config                                    |
|-------------------------|----------------|-----------------|-------------------------------------------|
| YOLOV3_1           | O              | O               | [YOLOV3_1.json](../../../test/data/YOLOV3_1.json) |
| YOLOV4_3          | O              | X               | [YOLOV4_3.json](../../../test/data/YOLOV4_3.json) |
| YOLOV5Pose640_1    | O              | X               | [YOLOV5Pose640_1.json](../../../test/data/YOLOV5Pose640_1.json) |
| YOLOV5S_1          | O              | O               | [YOLOV5S_1.json](../../../test/data/YOLOV5S_1.json) |
| YOLOV5S_3         | O              | O               | [YOLOV5S_3.json](../../../test/data/YOLOV5S_3.json) |
| YOLOV5S_4          | O              | O               | [YOLOV5S_4.json](../../../test/data/YOLOV5S_4.json) |
| YOLOV5S_6          | O              | O               | [YOLOV5S_6.json](../../../test/data/YOLOV5S_6.json) |
| YOLOV5S_Face-1     | O              | X               | [YOLOV5S_Face-1.json](../../../test/data/YOLOV5S_Face-1.json) |
| YOLOV5X_2          | O              | O               | [YOLOV5X_2.json](../../../test/data/YOLOV5X_2.json) |
| YOLOv7_512         | O              | O               | [YOLOV7_512.json](../../../test/data/YOLOV7_512.json) |
| YoloV7             | O              | O               | [YoloV7.json](../../../test/data/YoloV7.json) |
| YoloV8N            | O              | X               | [YoloV8N.json](../../../test/data/YoloV8N.json) |
| YOLOV9S            | O              | X               | [YOLOV9S.json](../../../test/data/YOLOV9S.json) |
| YOLOX-S_1          | O              | X               | [YOLOX-S_1.json](../../../test/data/YOLOX-S_1.json) |

Preconfigured JSON config files for supported YOLO models are available in the [test/data](../../../test/data) directory.  

**Alternative for unsupported cases**: For models or configurations not supported by **`YoloPostProcess`**, you can refer to the [`lib/pybind`](../../../lib/pybind) codes to implement custom C++ post-processing code and wrap it with Pybind11. This allows you to create optimized post-processing tailored to the specific characteristics of your model.

### Run  YoloPostProcess Python Example  

The following example demonstrates how to perform YOLO post-processing using the **`YoloPostProcess`** class from the **`dx_postprocess`** module.
```
$ python template/python/yolo_pybind_example.py --video_path /path/to/your/video_file --config_path /path/to/your/config_file --run_async --visualize
```

**Quick Example**:
```
python templates/python/yolo_pybind_example.py --video_path assets/videos/dance-group.mov --config test/data/YoloV7.json --run_async --visualize
```

This script takes a video file as input and performs Pre-processing, Inference, and Post-processing (synchronous or asynchronous).  

After processing all frames, the average FPS (Frames Per Second) is calculated as:  
**FPS = Total Frames / Total Processing Time (in seconds)**

**Command-Line Arguments**  

- `--video_path`: Path to the input video file  
- `--config_path`: Path to the JSON config file containing model path and post-processing parameters  
- `--run_async`: Use asynchronous inference with `RunAsync()`, where post-processing is performed inside the callback function. If **not** specified, synchronous inference with `Run()` is used, and preprocessing, inference, and post-processing are executed sequentially for each frame.  
- `--visualize`: Enables visualization of detection results  

**Post-Processing with `YoloPostProcess` Class**  

- **1.** Import the Module  
```
from dx_postprocess import YoloPostProcess
```

- **2.** Initialize the Post-Processor  
Pass the config file path to initialize YoloPostProcess.  
```
ypp = YoloPostProcess(json_config)
```

This sets up model-specific decoding parameters (e.g., class count, anchor sizes, thresholds).  

**For Synchronous Inference Mode**  
When `--run_async` is **not** specified,  

- Pre-processing → Inference → Post-processing  
- All steps run sequentially for each frame using `Run()`  

```
# Initialize InferenceEngine
ie = InferenceEngine(json_config["model"]["path"])

# Pre-Processing
input_tensor, _, _ = letter_box(frame, (input_size, input_size), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)

# Run the inference engine synchronously
ie_output = ie.Run(input_tensor)

# Post-Processing
pp_output = ypp.Run(ie_output)
```

**For Asynchronous Inference Mode**  
When `--run_async` is specified,  

- Uses `RunAsync()`  
- Post-processing occurs inside the callback function, allowing pipelined execution and improved throughput  

```
def pp_callback(ie_outputs, user_args):
    value:UserArgs = user_args.value
    pp_output_ = value.ypp_.Run(ie_outputs)
    q.put([value.input_tensor_, pp_output_])

class UserArgs:
    def __init__(self, ypp:YoloPostProcess, input_tensor):
        self.ypp_ = ypp
        self.input_tensor_ = input_tensor

# Initialize InferenceEngine
ie = InferenceEngine(json_config["model"]["path"])

# Register callback function
ie.RegisterCallBack(pp_callback)

# UserArgs for callback function
user_args = UserArgs(ypp, input_tensor)

# Pre-Processing
input_tensor, _, _ = letter_box(frame, (input_size, input_size), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)

# Run inference asynchronously
req_id = ie.RunAsync(input_tensor, user_args)
```

---


