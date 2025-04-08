## Overview
Post-processing for YOLO models can be easily implemented using Python libraries such as `numpy` and `torchvision.ops.nms`, as demonstrated in the `/template/python/yolov5s_example.py` and `/template/python/yolov_async.py` python example.

However, from a post-processing latency perspective, these implementations do not always guarantee optimal performance.

For real-time applications, minimizing post-processing latency is crucial for improving overall FPS (Frames Per Second). One effective approach to reducing post-processing latency is to implement optimized C++ post-processing code and wrap it using `Pybind11` for Python integration.

The typical steps in the YOLO post-processing include:

- Score filtering
- Bounding box decoding
- NMS (Non-Maximum Suppression)

Since these steps involve heavy matrix operations and iterative loops, executing them in C++ is significantly more efficient than in Python.

In `DX-APP`, we provide the `dx_postprocess` module, which includes the `YoloPostProcess` class—a `Pybind11`-wrapped C++ implementation of YOLO post-processing. Using the `YoloPostProcess` class, various YOLO models can be post-processed with a single function call.

## Supported Use Cases for `YoloPostProcess`
The specific details of YOLO post-processing may vary depending on the model type. Even with the same version of the YOLO model, the process may differ based on:

- Whether `PPU` (Post-Processing Unit) is applied during model compilation via `DX-COM`
- Whether `DX-RT` is built with `USE_ORT=ON`

For user convenience, the `Run()` function in the `YoloPostProcess` class automatically handles these different cases.

If `PPU` is enabled, all models (regardless of type: `BBOX`, `FACE`, `POSE`) can be post-processed using `YoloPostProcess`.

For models without `PPU`, the availability of post-processing depends on whether `USE_ORT=ON` is set during `DX-RT` compilation. The table below shows the details::

| Model  | USE_ORT = ON | USE_ORT = OFF |
| :----: | :----------: | :-----------: |
| YOLOv5 |      ✔️       |       ✔️       |
| YOLOv7 |      ✔️       |       ✔️       |
| YOLOX  |      ✔️       |       ✔️       |
| YOLOv8 |      ✔️       |       ✖️       |
| YOLOv9 |      ✔️       |       ✖️       |

If `USE_ORT=OFF`, post-processing for YOLOv8 and YOLOv9 is currently not supported in `YoloPostProcess`. Future support is planned. In the meantime, users can manually implement post-processing by modifying the `ProcessRAW()` function in `lib/pybind/yolo_post_processing.cpp`.

## Run `YoloPostProcess` Python Example
The following example demonstrates how to perform YOLO post-processing using the `YoloPostProcess` class:

```shell
$ python template/python/yolo_pybind_example.py --video_path /path/to/your/video_file --config_path /path/to/your/config_file --run_async --visualize
```
This script takes a video file as input, performs pre-processing, inference, and post-processing on each frame. After processing all frames, the overall `FPS` is computed as: 

$$
\text{FPS} = \frac{\text{Total Processing Time}}{\text{Total Frames}}
$$
### Command-Line Arguments
- `--video_path` : Path to the input video file
- `--config_path` : Path to the JSON config file containing model path and post-processing parameters
- `--run_async` : Use asynchronous inference with `RunAsync()`, where post-processing is performed inside the callback function. If not specified, synchronous inference with `Run()` is used, and preprocessing, inference, and post-processing are executed sequentially for each frame.
- `--visualize` : Enables visualization of detection results

Preconfigured JSON config files for supported YOLO models are available in the `./test` directory.

###  Post-Processing with `YoloPostProcess` Class
First, import the `YoloPostProcess` class from the `dx_postprocess` module:
```python
from dx_postprocess import YoloPostProcess
```
When initializing the `YoloPostProcess` class, pass the JSON config as an argument to set the necessary parameters for Post-Processing:
```python
ypp = YoloPostProcess(json_config)
```
**For Synchronous Inference Mode:**

When `--run_async` is not specified, synchronous inference with `Run()` is used, and preprocessing, inference, and post-processing are executed sequentially for each frame.
```python
# Initialize InferenceEngine
ie = InferenceEngine(json_config["model"]["path"])

# Pre-Processing
input_tensor, _, _ = letter_box(frame, (input_size, input_size), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)

# Run the inference engine synchronously
ie_output = ie.Run(input_tensor)

# Post-Processing
pp_output = ypp.Run(ie_output)
```
**For Asynchronous Inference Mode:**

When `--run_async` is specified, asynchronous inference with `RunAsync()` is used, where post-processing is performed inside the callback function.
```python
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