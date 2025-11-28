This chapter describes how to use the classification template to run image classification or regression models on DEEPX NPU. This workflow does **not** require source code modification, relying primarily on a **JSON configuration file** for setup.

!!! note "IMPORTANT"

    It is important to fully understand the output structure of your model, as output shapes may vary depending on model type and NPU compilation settings.   

**Output Post-Processing Recommendation**  
For faster result interpretation, it is recommended to include an ArgMax layer at the end of the model to directly output class indices.  

---

## Classification Template Example

This section explains how to execute an image classification task using a DXNN-compiled model and a JSON configuration file.  

**Configuration File**  
The template uses a JSON configuration file to define input/output settings and runtime parameters.  

- Sample Location: `example/run_classifier/imagenet_example.json`.  

**How to Run**  

To execute the demo using the configuration file, run the following command  

```
./bin/run_classifier -c example/run_classifier/imagenet_example.json
...
[sample/ILSVRC2012/0.jpeg] Top1 Result : class 905 (window shade)
[sample/ILSVRC2012/1.jpeg] Top1 Result : class 321 (admiral)
[sample/ILSVRC2012/2.jpeg] Top1 Result : class 846 (table lamp)
[sample/ILSVRC2012/3.jpeg] Top1 Result : class 794 (shower curtain)
...
```

**JSON Customization**  

No source code changes are required. The entire configuration is driven by the JSON file, allowing you to modify:

- Input dimensions, normalization, or format.
- Output processing parameters.
- Result display or file-based result saving configurations.

---

## Classification Post-Processing

### Using ArgMax Layer

For classification models where only the Top-1 prediction is required, it is recommended to include an ArgMax layer at the end of the model.

**Example Models**: EfficientNetB0_4 is configured with an ArgMax layer.

**Output Format (ArgMax)**
When using an ArgMax-based model, the NPU output is significantly simplified.

- The NPU output consists of a single `uint16_t` (2 bytes) value.  
- This value directly represents the class index with the highest probability.  

```
Task[0] npu_0, NPU, 8209728bytes (input 157696, output 2)
  inputs
    data, INT8, [1, 224, 224, 3, ], 0
  outputs
    argmax_output, UINT16, [1, ], 0

```

This value can be used as follows.

```
auto outputs = ie.Run(input_tensor);
auto max_index = *(uint16_t*)outputs.front()->data();
std::cout << "argmax : " << max_index << std::endl;
```

###  Using GAP or FC Layer

If the model's last layer is **not** ArgMax but a GAP (Global Average Pooling) or FC (Fully Connected) layer, the output is expected to be in the format `[1, 1, 1, number of classes]`.

- **Logic Required**: The host application must apply **Softmax** on the valid portion of this output to derive class probabilities.
- **Reference**: `lib/post_process/classification_post_processing.hpp` - Line 65  

```
#include "post_process/classification_post_processing.hpp"
  .
  .
  .
  // n : number of classes
  auto outputs = ie.Run(input_tensor);
  std::vector<float> scores = classification::getSoftmax((float*)outputs.back()->data(), n);
```

---
