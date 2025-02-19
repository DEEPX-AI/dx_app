# Run ImageNet Python Example

**Getting the usage of executable, Try run with "--help" option.**

- **Classification**

```shell
$ python template/python/imageNet_example.py
```

or

```shell
$ python template/python/imageNet_example.py --config example/imagenet_example.json
  ...
  [example/ILSVRC2012/0.jpeg] Top1 Result : class 831 (studio couch, day bed)
  [example/ILSVRC2012/1.jpeg] Top1 Result : class 321 (admiral)
  [example/ILSVRC2012/2.jpeg] Top1 Result : class 846 (table lamp)
  [example/ILSVRC2012/3.jpeg] Top1 Result : class 794 (shower curtain)
  Profiler data has been written to profiler.json
  Task0 , npu_0 : latency 1848 us, inference time 1411.75 us
  Device 0 : 4
```

Import The InferenceEngine module which callable inference engine module.

```python
  from dx_engine import InferenceEngine
```

Enter the model file as a parameter to **InferecneEngine** module.

```python
  ie = InferenceEngine("./example/EfficientNetB0_4/EfficientNetB0_4.dxnn")
```

The dxrt model has input and output tensors that shapes are N H W C format by default. Input tensor data format of current devices is aligned on 64-byte.  
 You should refer to the [Here](python/imageNet_example.py) and re-arrange input data.

```python
  def preprocessing(image, new_shape=(224, 224), align=64, format=None):
      image = cv2.resize(image, new_shape)
      h, w, c = image.shape
      if format is not None:
          image = cv2.cvtColor(image, format)
      if align == 0 :
          return image
      length = w * c
      align_factor = align - (length - (length & (-align)))
      image = np.reshape(image, (h, w * c))
      dummy = np.full([h, align_factor], 0, dtype=np.uint8)
      image_input = np.concatenate([image, dummy], axis=-1)

      return image_input
```

# Run YoloV5S Python Example

**Getting the usage of executable, Try run with "--help" option.**

- **Object Detection**

```shell
$ python template/python/yolov5s_example.py
```

or

```shell
$ python template/python/yolov5s_example.py --config example/yolov5s3_example.json
  ...
  [Result] Detected 10 Boxes.
  [0] conf, classID, x1, y1, x2, y2, : 0.8771, person(0), 307, 139, 401, 364
  [1] conf, classID, x1, y1, x2, y2, : 0.7358, bowl(45), 46, 317, 107, 347
  [2] conf, classID, x1, y1, x2, y2, : 0.7192, bowl(45), 25, 360, 79, 393
  [3] conf, classID, x1, y1, x2, y2, : 0.6766, oven(69), 0, 218, 154, 325
  [4] conf, classID, x1, y1, x2, y2, : 0.5811, oven(69), 389, 246, 497, 359
  [5] conf, classID, x1, y1, x2, y2, : 0.5664, person(0), 0, 295, 48, 332
  [6] conf, classID, x1, y1, x2, y2, : 0.5365, bowl(45), 1, 329, 69, 380
  [7] conf, classID, x1, y1, x2, y2, : 0.4199, potted plant(58), 0, 86, 50, 206
  [8] conf, classID, x1, y1, x2, y2, : 0.3649, bottle(39), 172, 271, 203, 323
  [9] conf, classID, x1, y1, x2, y2, : 0.3084, cup(41), 117, 300, 137, 327
  ...
```

After converting, You can easily cpu-post-processing using cpu_0.onnx output file with dxnn file.  
 Modify python application with reference to [templates/python/yolov5s_example.py](./python/yolov5s_example.py).

  <p align="center">
    <img src="./readme_images/result_python_yolov5s.jpg">
  </p>

Import The InferenceEngine module which callable inference engine module.

```python
  from dx_engine import InferenceEngine
```

Enter the model file as a parameter to **InferecneEngine** module.

```python
  ie = InferenceEngine("./example/YOLOV5S_3/YOLOV5S_3.dxnn")
```

In YOLO, the channel size of the feature map is typically calculated as (80 + 1 + 4) \* 3 = 255.
However, due to the characteristics of the NPU, if the channel size is less than 64 bytes, it is aligned to 16 bytes,
and if the channel size is 64 bytes or larger, it is aligned to 64 bytes.
As a result, each blob has 256 channels.  
 You should refer to the [Here](./python/yolov5s_example.py) and using all_decode function.

```python
      image_src = cv2.imread(input_path, cv2.IMREAD_COLOR)
      image_input, _, _ = letter_box(image_src, new_shape=(int(input_size), int(input_size)), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)

      ''' detect image (1) run dxrt inference engine, (2) post processing'''
      ie_output = ie.Run(image_input)
      print("dxrt inference Done! ")
      decoded_tensor = []
      if ie.output_dtype()[0] == "BBOX":
          decoded_tensor = ppu_decode(ie_output, layers)
      elif len(ie_output) > 1:
          cpu_model_path = os.path.join(os.path.split(model_path)[0], "cpu_0.onnx")
          if os.path.exists(cpu_model_path):
              decoded_tensor = onnx_decode(ie_output, cpu_model_path)
          else:
              decoded_tensor = all_decode(ie_output, layers)
      else:
          decoded_tensor = ie_output[0]
      print("decoding output Done! ")

      ''' post Processing '''
      x = np.squeeze(decoded_tensor)
      x = x[x[..., 4]>score_threshold]
      box = ops.xywh2xyxy(x[..., :4])
      x[:,5:] *= x[:,4:5]
      conf = np.max(x[..., 5:], axis=-1, keepdims=True)
      j = np.argmax(x[..., 5:], axis=-1, keepdims=True)
      mask = conf.flatten() > score_threshold
      filtered = np.concatenate((box, conf, j.astype(np.float32)), axis=1)[mask]
      sorted_indices = np.argsort(-filtered[:, 4])
      x = filtered[sorted_indices]
      x = torch.Tensor(x)
      x = x[torchvision.ops.nms(x[:,:4], x[:, 4], score_threshold)]

      ''' save result and print detected info '''
      print("[Result] Detected {} Boxes.".format(len(x)))
      image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
      colors = np.random.randint(0, 256, [80, 3], np.uint8).tolist()
      for idx, r in enumerate(x.numpy()):

          pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
          print("[{}] conf, classID, x1, y1, x2, y2, : {:.4f}, {}({}), {}, {}, {}, {}"
                .format(idx, conf, classes[label], label, pt1[0], pt1[1], pt2[0], pt2[1]))
          image = cv2.rectangle(image, pt1, pt2, colors[label], 2)
      cv2.imwrite("yolov5s.jpg", image)
      print("save file : yolov5s.jpg ")
```

# YOLO Post-Processing Optimization Using Pybind11
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
| YOLOv5 |      ✅       |       ✅       |
| YOLOv7 |      ✅       |       ✅       |
| YOLOX  |      ✅       |       ✅       |
| YOLOv8 |      ✅       |       ❌       |
| YOLOv9 |      ✅       |       ❌       |

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