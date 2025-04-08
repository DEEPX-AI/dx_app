This examples can be applied to both regression model and classification model.

It's essential to fully understand the model, as the output shape may differ depending on the NPU configuration.

- **Input Pre-Processing** 

Due to the characteristics of the M1 NPU, regardless of whether it is an OD model or an IC model, the input data’s channels must be a multiple of 64. 

- **Output Post-Processing**

Using ArgMax Layer in model. For more faster getting result value. 


## Run Classification Template

```shell
./bin/run_classifier -c example/imagenet_example.json
  ...
  [example/ILSVRC2012/0.jpeg] Top1 Result : class 831 (studio couch, day bed)
  [example/ILSVRC2012/1.jpeg] Top1 Result : class 321 (admiral)
  [example/ILSVRC2012/2.jpeg] Top1 Result : class 846 (table lamp)
  [example/ILSVRC2012/3.jpeg] Top1 Result : class 794 (shower curtain)
```

*example/imagenet_example.json* is a json config file that can run the classification model. Referring to this, you can customizing input and output.
And You can modify it in the application section for displaying or save classification results.

## Classification Pre-Processing

The method for converting the channels to a multiple of 64 is as follow example:

224 x 672 is represented as 224 x 224 x 3. To make the 672 channels a multiple of 64, 32 dummy channels are added.

Both general binary data and arrays (not just image files) need to be converted to a multiple of 64.

```cpp
#include "utils/common_util.hpp"
  .
  .
  .

  int alignFactor = dxapp::common::get_align_factor(width * channel, 64);
  uint8_t* input_tensor = new uint8_t[height * (width * channel + alignFactor)]
  /*
    if width=112, height=112, channel=3,
    alignFactor = (112 * 3) % 64 == 0 ? 0 : 64 - ((112 * 3) % 64)
    alignFactor = 48 
  */
  int copy_size = width * channel;
  for(int y=0; y<height; ++y)
  {
      memcpy(&input_tensor[y * (copy_size + alignFactor)],
              &src[y * copy_size], 
              copy_size
              );
  }
  auto outputs = ie.Run(input_tensor);
```

### When `USE_ORT=ON`

When using ONNX Runtime (`USE_ORT=ON`), there is no need to manually concatenate the dummy channels, as this operation is handled by ONNX Runtime.

However, since ONNX Runtime is being used, there may be some differences in speed performance.

Example : 

```cpp
  auto src = cv::imread(image_path, cv::IMREAD_COLOR);
  auto outputs = ie.Run(src.data);
```

## Classification Post-Processing

### Using ArgMax Layer in model

The EfficientNetB0_4 model included as an example is an ArgMax model.   

For an ARGMAX model where only the Top 1 result is needed, the NPU output will consist of a single `uint16_t` value (2 bytes).   
```console
    Task[0] npu_0, NPU, 8209728bytes (input 157696, output 2)
      inputs
        data, INT8, [1, 224, 224, 3,  ], 0
      outputs
        argmax_output, UINT16, [1,  ], 0
```

This value can be used as follows:
```cpp
  auto outputs = ie.Run(input_tensor);
  auto max_index = *(uint16_t*)outputs.front()->data();
  std::cout << "argmax : " << max_index << std::endl;
```

### When Using Global Average Pooling (GAP) or Fully Connected Layer (FC Layer)

If the model's last layer is not ArgMax but a GAP (Global Average Pooling) or FC (Fully Connected) layer, 
the output is expected to be in the format `[1, 1, 1, number of classes]`.

However, due to the characteristics of the M1A NPU, **if the output data channels are greater than 64, 
they are aligned to a multiple of 64. 
If the channels are smaller than 64, they are aligned to a multiple of 16**.

Example:

- ONNX model output: `1 x 1 x 1 x 1000` → NPU output: `1 x 1 x 1 x 1024` (dummy: 24 bytes)
- ONNX model output: `1 x 1 x 1 x 10` → NPU output: `1 x 1 x 1 x 16` (dummy: 6 bytes)
- ONNX model output: `1 x 1 x 1 x 30` → NPU output: `1 x 1 x 1 x 32` (dummy: 2 bytes)

Therefore, the confidence must be calculated using Softmax on the CPU. Below is an example code snippet.   
Reference: lib/post_process/classification_post_processing.hpp, line 65 
```cpp
#include "post_process/classification_post_processing.hpp"
  .
  .
  .
  // n : number of classes
  auto outputs = ie.Run(input_tensor);
  std::vector<float> scores = classification::getSoftmax((float*)outputs.back()->data(), n);
```