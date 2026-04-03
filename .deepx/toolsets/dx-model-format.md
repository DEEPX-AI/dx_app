# .dxnn Model Format Reference

> The compiled neural network model format for DEEPX NPU inference.

## Overview

`.dxnn` is the compiled binary model format used by DEEPX NPU devices. It contains
optimized neural network weights, graph structure, and tensor specifications compiled
for direct execution on the DX-M1/DX-M1A hardware.

## File Description

| Property | Value |
|---|---|
| Extension | `.dxnn` |
| Type | Binary (not human-readable) |
| Format version | v7+ (dx_app v3.0.0) |
| Typical size | 2-50 MB depending on model |
| Contains | Compiled graph, quantized weights, tensor specs, metadata |

A `.dxnn` file is self-contained: it includes everything the NPU needs to execute
inference without external dependencies.

## Input/Output Tensor Specifications

### Input Tensors

| Property | Description |
|---|---|
| Layout | NHWC (batch, height, width, channels) or NCHW |
| Data type | UINT8 (quantized) or FP16 (half-precision) |
| Batch dimension | Fixed at compilation time (typically N=1) |
| Spatial dimensions | Fixed (e.g., 640x640, 320x320, 224x224) |
| Channels | 3 (RGB/BGR) for vision models |

### Output Tensors

| Property | Description |
|---|---|
| Count | 1 or more output heads per model |
| Data type | INT8, UINT8, or FP16 depending on compilation |
| Layout | Task-dependent (see below) |

### Output Layouts by Task

| Task | Typical Output Shape | Description |
|---|---|---|
| object_detection | `(1, N, 4+1+C)` | N proposals, 4 bbox + 1 obj + C classes |
| classification | `(1, C)` | C class scores |
| pose_estimation | `(1, N, 4+1+K*3)` | N persons, bbox + obj + K keypoints (x,y,conf) |
| instance_segmentation | `(1, N, 4+1+C+32)`, `(1, 32, H, W)` | Proposals + mask prototypes |
| semantic_segmentation | `(1, C, H, W)` | Per-pixel class logits |
| face_detection | `(1, N, 4+1+10)` | N faces, bbox + score + 5 landmarks |
| depth_estimation | `(1, 1, H, W)` | Per-pixel depth values |
| image_denoising | `(1, 3, H, W)` or `(1, H, W, 3)` | Restored image |
| image_enhancement | `(1, 3, H, W)` | Enhanced image |
| super_resolution | `(1, 3, H*S, W*S)` | Upscaled image (S = scale factor) |
| embedding | `(1, D)` | D-dimensional feature vector |
| obb_detection | `(1, N, 4+1+1+C)` | N proposals with rotation angle |
| hand_landmark | `(1, 63)` | 21 landmarks * 3 (x, y, z) |
| ppu | `(1, N, 4+1+C)` | PPU-specific detection format |

## Data Types

| Type | Bits | Range | Use Case |
|---|---|---|---|
| INT8 | 8 | -128 to 127 | Quantized weights and activations |
| UINT8 | 8 | 0 to 255 | Quantized activations (unsigned) |
| FP16 | 16 | ±65504 | Half-precision for sensitive layers |

Most `.dxnn` models use INT8 quantization for maximum throughput. The dx-compiler
applies post-training quantization (PTQ) or quantization-aware training (QAT) data
during compilation.

## Compilation Flow

```
Source Model          dx-compiler             NPU Binary
============         ===========             ==========

ONNX (.onnx)    ---> Graph optimization  ---> .dxnn
PyTorch (.pt)   ---> Quantization (PTQ)
TensorFlow (.pb)     Layer fusion
TFLite (.tflite)     Memory planning
                     Code generation
```

### Step-by-Step

1. **Export to ONNX** — Convert source model (PyTorch, TensorFlow, etc.) to ONNX format
2. **Compile with dx-compiler** — Run the compiler to produce `.dxnn`:
   ```bash
   dx-compiler --input model.onnx \
               --output model.dxnn \
               --target dx_m1 \
               --quantize int8 \
               --calibration-data cal_data/ \
               --input-shape 1,3,640,640
   ```
3. **Validate** — Verify with `dxrt-cli`:
   ```bash
   dxrt-cli --info model.dxnn
   ```

### Compiler Options

| Option | Description |
|---|---|
| `--input` | Source model path (ONNX, TFLite) |
| `--output` | Output `.dxnn` path |
| `--target` | NPU target (`dx_m1`, `dx_m1a`) |
| `--quantize` | Quantization mode (`int8`, `fp16`, `mixed`) |
| `--calibration-data` | Directory of representative input images for PTQ |
| `--input-shape` | Input tensor shape (NCHW format) |
| `--batch-size` | Batch size to compile for (default: 1) |
| `--optimize` | Optimization level (0-3, default: 2) |

## Format Version History

| Version | DX-RT Compat | Features |
|---|---|---|
| v7 | 3.0.x | Multi-output, profiling metadata, extended quantization |
| v6 | 2.5.x | Batch inference support |
| v5 | 2.0.x | Basic single-output models |
| v4 | 1.x | Legacy format — not supported in dx_app v3.0.0 |

**Important:** `.dxnn` files compiled with v5 or v6 format are NOT guaranteed to work
with DX-RT 3.0.x. Always recompile models when upgrading DX-RT. See
`memory/common_pitfalls.md` [UNIVERSAL] entry on version mismatch.

## Tensor Layout: NCHW vs NHWC

| Layout | Convention | When Used |
|---|---|---|
| NCHW | (batch, channels, height, width) | Default compilation layout |
| NHWC | (batch, height, width, channels) | Some models, OpenCV compatibility |

The dx-compiler embeds the layout in the `.dxnn` file. `InferenceEngine` handles
layout internally — users provide input in the format expected by `get_input_shape()`.

```python
shape = engine.get_input_shape()
# shape = (1, 640, 640, 3)  means NHWC
# shape = (1, 3, 640, 640)  means NCHW
```

## File Inspection

```bash
# View model metadata
dxrt-cli --info model.dxnn
# Output:
#   Model: yolov8n
#   Format: v7
#   Target: dx_m1
#   Input: [1, 640, 640, 3] UINT8
#   Output 0: [1, 8400, 84] FP16
#   Quantization: INT8 (PTQ)
#   Size: 6.2 MB
#   Compiled: 2025-01-15

# Verify file integrity
dxrt-cli --verify model.dxnn
```

## Storage and Download

Models are stored in a central repository and downloaded via `setup.sh`:

```bash
# Download all supported models
./setup.sh

# Models are placed in the models/ directory
ls models/
# yolov8n.dxnn  yolov5n.dxnn  efficientnet_b0.dxnn  ...
```

The `model_registry.json` file lists all available models and their metadata.
Only models with `"supported": true` can be downloaded via `setup.sh`.
