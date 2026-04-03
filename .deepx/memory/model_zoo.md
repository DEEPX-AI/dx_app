# Model Zoo — dx_app

> 133 models across 15 AI tasks with metadata, performance tiers, and recommendations.

## Overview

dx_app v3.0.0 supports 133 compiled `.dxnn` models organized into 15 AI tasks.
All model metadata is stored in `config/model_registry.json`. This document provides
a curated reference with representative models, performance tiers, and recommended
postprocessors.

---

## Object Detection (~45 models)

### Representative Models

| Model | Input Size | Classes | Postprocessor | Tier |
|---|---|---|---|---|
| yolov5n | 640x640 | 80 (COCO) | YoloV5PostProcess | Fast |
| yolov5s | 640x640 | 80 | YoloV5PostProcess | Balanced |
| yolov5m | 640x640 | 80 | YoloV5PostProcess | Accurate |
| yolov7 | 640x640 | 80 | YoloV7PostProcess | Accurate |
| yolov7_tiny | 416x416 | 80 | YoloV7TinyPostProcess | Fast |
| yolov8n | 640x640 | 80 | YoloV8PostProcess | Fast |
| yolov8s | 640x640 | 80 | YoloV8PostProcess | Balanced |
| yolov8m | 640x640 | 80 | YoloV8PostProcess | Accurate |
| yolov10n | 640x640 | 80 | YoloV10PostProcess | Fast |
| yolov10s | 640x640 | 80 | YoloV10PostProcess | Balanced |
| yolov11n | 640x640 | 80 | YoloV11PostProcess | Fast |
| yolov11s | 640x640 | 80 | YoloV11PostProcess | Balanced |
| yolo26n | 640x640 | 80 | Yolo26PostProcess | Fast |
| yolo26s | 640x640 | 80 | Yolo26PostProcess | Balanced |
| ssd_mobilenetv2 | 300x300 | 91 | SSDPostProcess | Fast |
| efficientdet_d0 | 512x512 | 90 | EfficientDetPostProcess | Balanced |
| nanodet_m | 320x320 | 80 | NanodetPostProcess | Ultra-Fast |

### Output Format

Detection models output `(1, N, 4+1+C)` tensors where:
- N = number of proposals (varies: 8400 for v8, 25200 for v5)
- 4 = bbox coordinates (x, y, w, h or x1, y1, x2, y2)
- 1 = objectness score
- C = class scores (80 for COCO)

### Recommended Starting Points

- **Fastest inference:** nanodet_m (320x320) or yolov5n
- **Best accuracy/speed tradeoff:** yolov8s or yolo26s
- **Highest accuracy:** yolov8m or yolov7

---

## Classification (~20 models)

| Model | Input Size | Classes | Postprocessor | Tier |
|---|---|---|---|---|
| efficientnet_b0 | 224x224 | 1000 (ImageNet) | EfficientNetPostProcess | Fast |
| efficientnet_b1 | 240x240 | 1000 | EfficientNetPostProcess | Balanced |
| efficientnet_b4 | 380x380 | 1000 | EfficientNetPostProcess | Accurate |
| mobilenetv2 | 224x224 | 1000 | MobileNetV2PostProcess | Fast |
| resnet18 | 224x224 | 1000 | ResNetPostProcess | Fast |
| resnet50 | 224x224 | 1000 | ResNetPostProcess | Balanced |
| resnet101 | 224x224 | 1000 | ResNetPostProcess | Accurate |
| inception_v3 | 299x299 | 1000 | ClassificationPostProcess | Balanced |
| squeezenet | 224x224 | 1000 | ClassificationPostProcess | Ultra-Fast |

### Output Format

Classification models output `(1, C)` where C = number of classes.
Postprocessors apply softmax and return top-K results.

---

## Pose Estimation (~8 models)

| Model | Input Size | Keypoints | Postprocessor | Tier |
|---|---|---|---|---|
| yolov5s_pose | 640x640 | 17 (COCO) | YoloV5PosePostProcess | Balanced |
| yolov5m_pose | 640x640 | 17 | YoloV5PosePostProcess | Accurate |
| yolov8n_pose | 640x640 | 17 | YoloV8PosePostProcess | Fast |
| yolov8s_pose | 640x640 | 17 | YoloV8PosePostProcess | Balanced |

### Output Format

Pose models output `(1, N, 4+1+K*3)` where K=17 COCO keypoints.
Each keypoint has (x, y, confidence).

---

## Instance Segmentation (~10 models)

| Model | Input Size | Classes | Postprocessor | Tier |
|---|---|---|---|---|
| yolov5n_seg | 640x640 | 80 | YoloV5SegPostProcess | Fast |
| yolov5s_seg | 640x640 | 80 | YoloV5SegPostProcess | Balanced |
| yolov8n_seg | 640x640 | 80 | YoloV8SegPostProcess | Fast |
| yolov8s_seg | 640x640 | 80 | YoloV8SegPostProcess | Balanced |

### Output Format

Instance segmentation outputs two tensors:
- `(1, N, 4+1+C+32)` — proposals with mask coefficients
- `(1, 32, H/4, W/4)` — mask prototypes

---

## Semantic Segmentation (~10 models)

| Model | Input Size | Classes | Postprocessor | Tier |
|---|---|---|---|---|
| bisenetv1 | 1024x512 | 19 (Cityscapes) | BisenetV1PostProcess | Fast |
| deeplabv3plus_mobilenet | 513x513 | 21 (VOC) | DeepLabV3PostProcess | Balanced |
| deeplabv3plus_resnet | 513x513 | 21 | DeepLabV3PostProcess | Accurate |
| segformer_b0 | 512x512 | 19 | ClassificationPostProcess | Fast |
| segformer_b1 | 512x512 | 19 | ClassificationPostProcess | Balanced |

### Output Format

Semantic segmentation outputs `(1, C, H, W)` — per-pixel class logits.
Argmax along class dimension gives the segmentation mask.

---

## Face Detection (~8 models)

| Model | Input Size | Landmarks | Postprocessor | Tier |
|---|---|---|---|---|
| scrfd_10g | 640x640 | 5 | SCRFDPostProcess | Accurate |
| scrfd_2.5g | 640x640 | 5 | SCRFDPostProcess | Fast |
| yolov5s_face | 640x640 | 5 | YoloV5FacePostProcess | Balanced |
| yolov5n_face | 640x640 | 5 | YoloV5FacePostProcess | Fast |
| retinaface | 640x640 | 5 | RetinaFacePostProcess | Balanced |

### Output Format

Face detection outputs `(1, N, 4+1+10)` where 10 = 5 landmarks * 2 (x, y).

---

## Depth Estimation (~4 models)

| Model | Input Size | Output | Postprocessor | Tier |
|---|---|---|---|---|
| fastdepth_1 | 224x224 | Depth map | FastDepthPostProcess | Fast |
| fastdepth_2 | 224x224 | Depth map | FastDepthPostProcess | Balanced |

### Output Format

Depth models output `(1, 1, H, W)` — per-pixel depth in meters.

---

## Image Denoising (~6 models)

| Model | Input Size | Noise Level | Postprocessor | Tier |
|---|---|---|---|---|
| dncnn_15 | 481x481 | sigma=15 | DnCNNPostProcess | Light noise |
| dncnn_25 | 481x481 | sigma=25 | DnCNNPostProcess | Medium noise |
| dncnn_50 | 481x481 | sigma=50 | DnCNNPostProcess | Heavy noise |

---

## Image Enhancement (~3 models)

| Model | Input Size | Task | Postprocessor | Tier |
|---|---|---|---|---|
| zero_dce | 256x256 | Low-light | ZeroDCEPostProcess | Fast |
| zero_dce_pp | 256x256 | Low-light | ZeroDCEPostProcess | Balanced |

---

## Super Resolution (~4 models)

| Model | Input Size | Scale | Postprocessor | Tier |
|---|---|---|---|---|
| espcn_x2 | 224x224 | 2x | ESPCNPostProcess | Fast |
| espcn_x3 | 224x224 | 3x | ESPCNPostProcess | Balanced |
| espcn_x4 | 224x224 | 4x | ESPCNPostProcess | Balanced |

---

## Embedding (~3 models)

| Model | Input Size | Dim | Postprocessor | Tier |
|---|---|---|---|---|
| arcface_mobilefacenet | 112x112 | 128 | ArcFacePostProcess | Fast |
| arcface_resnet50 | 112x112 | 512 | ArcFacePostProcess | Accurate |

---

## OBB Detection (~3 models)

| Model | Input Size | Classes | Postprocessor | Tier |
|---|---|---|---|---|
| yolo26n_obb | 640x640 | 15 (DOTA) | Yolo26OBBPostProcess | Fast |
| yolo26s_obb | 640x640 | 15 | Yolo26OBBPostProcess | Balanced |

**Note:** OBB uses score_threshold only — no NMS threshold.

---

## Hand Landmark (~3 models)

| Model | Input Size | Landmarks | Postprocessor | Tier |
|---|---|---|---|---|
| handlandmarklite_1 | 224x224 | 21 | HandLandmarkPostProcess | Fast |
| handlandmarklite_2 | 224x224 | 21 | HandLandmarkPostProcess | Balanced |

---

## PPU (~6 models)

| Model | Input Size | Classes | Postprocessor | Tier |
|---|---|---|---|---|
| yolov5s_ppu | 640x640 | 80 | PPUPostProcess | Balanced |
| yolov7_ppu | 640x640 | 80 | PPUPostProcess | Balanced |
| yolov8n_ppu | 640x640 | 80 | PPUPostProcess | Fast |

**Important:** PPU models use dedicated `PPUPostProcess`. Do NOT use standard YOLO
postprocessors with PPU models.

---

## Performance Tiers

| Tier | Description | Typical FPS (DX-M1, 640x640) |
|---|---|---|
| Ultra-Fast | Minimal latency, small models | 100+ fps |
| Fast | Low latency, nano variants | 60-100 fps |
| Balanced | Good accuracy/speed tradeoff | 30-60 fps |
| Accurate | Highest accuracy, larger models | 10-30 fps |

**Note:** FPS varies by input resolution, postprocess complexity, and whether using
sync or async runner. Async typically achieves 2-3x the sync FPS.

## Choosing a Model

1. **Start with the Fast tier** — nano variants for rapid prototyping
2. **Move to Balanced** for production — small variants offer good accuracy
3. **Use Accurate only when needed** — medium/large variants for quality-critical apps
4. **Check `supported: true`** in model_registry.json before selecting
