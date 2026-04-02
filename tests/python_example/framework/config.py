from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Tuple, Union

SRC_ROOT = Path(__file__).parent.parent.parent.parent / "src" / "python_example"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    DEPTH_ESTIMATION = "depth_estimation"
    IMAGE_RESTORATION = "image_restoration"
    IMAGE_ENHANCEMENT = "image_enhancement"
    SUPER_RESOLUTION = "super_resolution"

@dataclass
class ModelConfig:
    name: str
    task: TaskType
    base_path: str
    class_name: str
    model_filename: Optional[str] = None

    ort_on_output_shapes: Optional[List[Tuple]] = None
    ort_off_output_shapes: Optional[List[Tuple]] = None
    postprocess_result_shape: Optional[Union[Tuple, List[Tuple]]] = None

    has_keypoints: bool = False
    num_keypoints: int = 0
    keypoint_dim: int = 2
    detection_output_size: int = 6

    @cached_property
    def model_path(self) -> Path:
        return SRC_ROOT / self.base_path

    @cached_property
    def variants(self) -> List[str]:
        if not self.model_path.exists():
            return []

        pattern = f"{self.name}_*.py"
        return sorted([f.name for f in self.model_path.glob(pattern) if f.is_file()])


EFFICIENTNET_CONFIG = ModelConfig(
    name="efficientnetb2",
    task=TaskType.CLASSIFICATION,
    base_path="classification/efficientnetb2",
    class_name="EfficientNetB2",
    model_filename="EfficientNetB2.dxnn",
    ort_on_output_shapes=[(1, 1000)],
    ort_off_output_shapes=[(1, 1000)],
)

SCRFD_CONFIG = ModelConfig(
    name="scrfd500m",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/scrfd500m",
    class_name="SCRFD",
    model_filename="SCRFD500M.dxnn",
    ort_on_output_shapes=[
        (1, 12800, 1),
        (1, 3200, 4),
        (1, 3200, 10),
        (1, 800, 10),
        (1, 3200, 1),
        (1, 800, 4),
        (1, 12800, 10),
        (1, 800, 1),
        (1, 12800, 4),
    ],
    has_keypoints=True,
    num_keypoints=5,
    keypoint_dim=2,
    detection_output_size=16,
    postprocess_result_shape=(1, 16),
)

YOLOX_CONFIG = ModelConfig(
    name="yoloxs",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yoloxs",
    class_name="YOLOX",
    model_filename="YoloXS.dxnn",
    ort_on_output_shapes=[(1, 5376, 85)],
    postprocess_result_shape=(1, 6),
)

YOLOV5FACE_CONFIG = ModelConfig(
    name="yolov5s_face",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/yolov5s_face",
    class_name="YOLOv5Face",
    model_filename="YOLOv5s_Face.dxnn",
    ort_on_output_shapes=[(1, 25200, 16)],
    has_keypoints=True,
    num_keypoints=5,
    keypoint_dim=2,
    detection_output_size=16,
    postprocess_result_shape=(1, 16),
)

YOLOV5POSE_CONFIG = ModelConfig(
    name="yolov5pose",
    task=TaskType.OBJECT_DETECTION,
    base_path="pose_estimation/yolov5pose",
    class_name="YOLOv5Pose",
    model_filename="YOLOV5Pose640_1.dxnn",
    ort_on_output_shapes=[(1, 25500, 57)],
    has_keypoints=True,
    num_keypoints=17,
    keypoint_dim=3,
    detection_output_size=57,
    postprocess_result_shape=(1, 57),
)

YOLOV5POSE_PPU_CONFIG = ModelConfig(
    name="yolov5pose_ppu",
    task=TaskType.OBJECT_DETECTION,
    base_path="ppu/yolov5pose_ppu",
    class_name="YOLOv5Pose_PPU",
    model_filename="YOLOV5Pose640_1.dxnn",
    ort_on_output_shapes=[(1, 1, 256)],
    has_keypoints=True,
    num_keypoints=17,
    keypoint_dim=3,
    detection_output_size=57,
    postprocess_result_shape=(1, 57),
)

YOLOV5_CONFIG = ModelConfig(
    name="yolov5s",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov5s",
    class_name="YOLOv5",
    model_filename="YoloV5S.dxnn",
    ort_on_output_shapes=[(1, 25200, 85)],
    ort_off_output_shapes=[
        (1, 255, 80, 80),
        (1, 255, 40, 40),
        (1, 255, 20, 20),
    ],
    postprocess_result_shape=(1, 6),
)

YOLOV5_PPU_CONFIG = ModelConfig(
    name="yolov5_ppu",
    task=TaskType.OBJECT_DETECTION,
    base_path="ppu/yolov5s_ppu",
    class_name="YOLOv5_PPU",
    model_filename="YoloV5S_PPU.dxnn",
    ort_on_output_shapes=[(1, 1, 32)],
    ort_off_output_shapes=[(1, 1, 32)],
    postprocess_result_shape=(1, 6),
)

YOLOV7_CONFIG = ModelConfig(
    name="yolov7",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov7",
    class_name="YOLOv7",
    model_filename="YoloV7.dxnn",
    ort_on_output_shapes=[(1, 25200, 85)],
    ort_off_output_shapes=[
        (1, 255, 80, 80),
        (1, 255, 40, 40),
        (1, 255, 20, 20),
    ],
    postprocess_result_shape=(1, 6),
)

YOLOV7_PPU_CONFIG = ModelConfig(
    name="yolov7_ppu",
    task=TaskType.OBJECT_DETECTION,
    base_path="ppu/yolov7_ppu",
    class_name="YOLOv7_PPU",
    model_filename="YoloV7_PPU.dxnn",
    ort_on_output_shapes=[(1, 1, 32)],
    ort_off_output_shapes=[(1, 1, 32)],
    postprocess_result_shape=(1, 6),
)


YOLOV8_CONFIG = ModelConfig(
    name="yolov8n",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov8n",
    class_name="YOLOv8",
    model_filename="YoloV8N.dxnn",
    ort_on_output_shapes=[(1, 84, 8400)],
    ort_off_output_shapes=[
        (1, 80, 80, 80),
        (1, 64, 80, 80),
        (1, 80, 40, 40),
        (1, 64, 40, 40),
        (1, 80, 20, 20),
        (1, 64, 20, 20),
    ],
    postprocess_result_shape=(1, 6),
)

YOLOV8SEG_CONFIG = ModelConfig(
    name="yolov8n_seg",
    task=TaskType.INSTANCE_SEGMENTATION,
    base_path="instance_segmentation/yolov8n_seg",
    class_name="YOLOv8Seg",
    model_filename="yolov8n_seg.dxnn",
    ort_on_output_shapes=[(1, 116, 8400),
                          (1, 32, 160, 160)],
    postprocess_result_shape=[(1, 6),
                              (1, 640, 640)]
)

YOLOV9_CONFIG = ModelConfig(
    name="yolov9s",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov9s",
    class_name="YOLOv9",
    model_filename="YoloV9S.dxnn",
    ort_on_output_shapes=[(1, 84, 8400)],
    ort_off_output_shapes=[
        (1, 80, 80, 80),
        (1, 64, 80, 80),
        (1, 80, 40, 40),
        (1, 64, 40, 40),
        (1, 80, 20, 20),
        (1, 64, 20, 20),
    ],
    postprocess_result_shape=(1, 6),
)

YOLOV10_CONFIG = ModelConfig(
    name="yolov10n",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov10n",
    class_name="YOLOv10",
    model_filename="YOLOV10N.dxnn",
    ort_on_output_shapes=[(1, 300, 6)],
    postprocess_result_shape=(1, 6),
)

YOLOV11_CONFIG = ModelConfig(
    name="yolov11n",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov11n",
    class_name="YOLOv11",
    model_filename="YOLOV11N.dxnn",
    ort_on_output_shapes=[(1, 84, 8400)],
    postprocess_result_shape=(1, 6),
)

YOLOV12_CONFIG = ModelConfig(
    name="yolov12",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov12n",
    class_name="YOLOv12",
    model_filename="YOLOV12N.dxnn",
    ort_on_output_shapes=[(1, 84, 8400)],
    postprocess_result_shape=(1, 6),
)

YOLOV26_CONFIG = ModelConfig(
    name="yolo26s",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolo26s",
    class_name="YOLOv26",
    model_filename="yolo26s.dxnn",
    ort_on_output_shapes=[(1, 300, 6)],
    postprocess_result_shape=(1, 6),
)

YOLOV26OBB_CONFIG = ModelConfig(
    name="yolov26obb",
    task=TaskType.OBJECT_DETECTION,
    base_path="obb_detection/yolo26n_obb",
    class_name="YOLOv26Obb",
    model_filename="yolo26s-obb.dxnn",
    detection_output_size = 7,
    ort_on_output_shapes=[(1, 300, 7)],
    postprocess_result_shape=(1, 7),
)

DEEPLABV3_CONFIG = ModelConfig(
    name="deeplabv3plusmobilenet",
    task=TaskType.SEMANTIC_SEGMENTATION,
    base_path="semantic_segmentation/deeplabv3plusmobilenet",
    class_name="DeepLabv3",
    model_filename="DeepLabV3PlusMobilenet.dxnn",
    ort_on_output_shapes=[(1, 19, 640, 640)],
    ort_off_output_shapes=[(1, 19, 640, 640)],
    postprocess_result_shape=(640, 640),
)

# ---------------------------------------------------------------------------
# Face Detection — additional variants
# ---------------------------------------------------------------------------
SCRFD2_5G_CONFIG = ModelConfig(
    name="scrfd2_5g",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/scrfd2_5g",
    class_name="SCRFD",
    model_filename="SCRFD2_5G.dxnn",
    ort_on_output_shapes=[
        (1, 12800, 1), (1, 3200, 4), (1, 3200, 10),
        (1, 800, 10), (1, 3200, 1), (1, 800, 4),
        (1, 12800, 10), (1, 800, 1), (1, 12800, 4),
    ],
    has_keypoints=True, num_keypoints=5, keypoint_dim=2,
    detection_output_size=16, postprocess_result_shape=(1, 16),
)

SCRFD10G_CONFIG = ModelConfig(
    name="scrfd10g",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/scrfd10g",
    class_name="SCRFD",
    model_filename="SCRFD10G.dxnn",
    ort_on_output_shapes=[
        (1, 12800, 1), (1, 3200, 4), (1, 3200, 10),
        (1, 800, 10), (1, 3200, 1), (1, 800, 4),
        (1, 12800, 10), (1, 800, 1), (1, 12800, 4),
    ],
    has_keypoints=True, num_keypoints=5, keypoint_dim=2,
    detection_output_size=16, postprocess_result_shape=(1, 16),
)

YOLOV5M_FACE_CONFIG = ModelConfig(
    name="yolov5m_face",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/yolov5m_face",
    class_name="YOLOv5Face",
    model_filename="YOLOv5m_Face.dxnn",
    ort_on_output_shapes=[(1, 25200, 16)],
    has_keypoints=True, num_keypoints=5, keypoint_dim=2,
    detection_output_size=16, postprocess_result_shape=(1, 16),
)

YOLOV7_FACE_CONFIG = ModelConfig(
    name="yolov7_face",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/yolov7_face",
    class_name="YOLOv7Face",
    model_filename="YoloV7_Face.dxnn",
    ort_on_output_shapes=[(1, 25200, 21)],
    has_keypoints=True, num_keypoints=5, keypoint_dim=2,
    detection_output_size=21, postprocess_result_shape=(1, 21),
)

YOLOV7S_FACE_CONFIG = ModelConfig(
    name="yolov7s_face",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/yolov7s_face",
    class_name="YOLOv7Face",
    model_filename="YoloV7s_Face.dxnn",
    ort_on_output_shapes=[(1, 25200, 21)],
    has_keypoints=True, num_keypoints=5, keypoint_dim=2,
    detection_output_size=21, postprocess_result_shape=(1, 21),
)

YOLOV7_W6_FACE_CONFIG = ModelConfig(
    name="yolov7_w6_face",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/yolov7_w6_face",
    class_name="YOLOv7Face",
    model_filename="YoloV7_W6_Face.dxnn",
    ort_on_output_shapes=[(1, 43008, 21)],
    has_keypoints=True, num_keypoints=5, keypoint_dim=2,
    detection_output_size=21, postprocess_result_shape=(1, 21),
)

YOLOV7_W6_TTA_FACE_CONFIG = ModelConfig(
    name="yolov7_w6_tta_face",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/yolov7_w6_tta_face",
    class_name="YOLOv7Face",
    model_filename="YoloV7_W6_TTA_Face.dxnn",
    ort_on_output_shapes=[(1, 43008, 21)],
    has_keypoints=True, num_keypoints=5, keypoint_dim=2,
    detection_output_size=21, postprocess_result_shape=(1, 21),
)

RETINAFACE_CONFIG = ModelConfig(
    name="retinaface_mobilenet0_25_640",
    task=TaskType.OBJECT_DETECTION,
    base_path="face_detection/retinaface_mobilenet0_25_640",
    class_name="RetinaFace",
    model_filename="RetinaFace_Mobilenet0_25_640.dxnn",
    # RetinaFace 640x640: bbox[1,16800,4] + cls[1,16800,2] + landmark[1,16800,10]
    ort_on_output_shapes=[(1, 16800, 4), (1, 16800, 2), (1, 16800, 10)],
    has_keypoints=True, num_keypoints=5, keypoint_dim=2,
    detection_output_size=16, postprocess_result_shape=(1, 16),
)

# ---------------------------------------------------------------------------
# Object Detection — size variants & new families
# ---------------------------------------------------------------------------
YOLOV5N_CONFIG = ModelConfig(
    name="yolov5n",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov5n",
    class_name="YOLOv5",
    model_filename="YoloV5N.dxnn",
    ort_on_output_shapes=[(1, 25200, 85)],
    postprocess_result_shape=(1, 6),
)

YOLOV5M_CONFIG = ModelConfig(
    name="yolov5m",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov5m",
    class_name="YOLOv5",
    model_filename="YoloV5M.dxnn",
    ort_on_output_shapes=[(1, 25200, 85)],
    postprocess_result_shape=(1, 6),
)

YOLOV5L_CONFIG = ModelConfig(
    name="yolov5l",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov5l",
    class_name="YOLOv5",
    model_filename="YoloV5L.dxnn",
    ort_on_output_shapes=[(1, 25200, 85)],
    postprocess_result_shape=(1, 6),
)

YOLOV5S_PPU_CONFIG = ModelConfig(
    name="yolov5s_ppu",
    task=TaskType.OBJECT_DETECTION,
    base_path="ppu/yolov5s_ppu",
    class_name="YOLOv5_PPU",
    model_filename="YoloV5S_PPU.dxnn",
    ort_on_output_shapes=[(1, 1, 32)],
    ort_off_output_shapes=[(1, 1, 32)],
    postprocess_result_shape=(1, 6),
)

YOLOV7TINY_CONFIG = ModelConfig(
    name="yolov7tiny",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov7tiny",
    class_name="YOLOv7",
    model_filename="YoloV7Tiny.dxnn",
    ort_on_output_shapes=[(1, 25200, 85)],
    postprocess_result_shape=(1, 6),
)

YOLOV8S_CONFIG = ModelConfig(
    name="yolov8s",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov8s",
    class_name="YOLOv8",
    model_filename="YoloV8S.dxnn",
    ort_on_output_shapes=[(1, 84, 8400)],
    postprocess_result_shape=(1, 6),
)

YOLOV9C_CONFIG = ModelConfig(
    name="yolov9c",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov9c",
    class_name="YOLOv9",
    model_filename="YoloV9C.dxnn",
    ort_on_output_shapes=[(1, 84, 8400)],
    postprocess_result_shape=(1, 6),
)

YOLO26N_CONFIG = ModelConfig(
    name="yolo26n",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolo26n",
    class_name="YOLOv26",
    model_filename="yolo26n.dxnn",
    ort_on_output_shapes=[(1, 300, 6)],
    postprocess_result_shape=(1, 6),
)

YOLOX_LEAKY_CONFIG = ModelConfig(
    name="yolox_s_leaky",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolox_s_leaky",
    class_name="YOLOX",
    model_filename="YoloxSLeaky.dxnn",
    ort_on_output_shapes=[(1, 5376, 85)],
    postprocess_result_shape=(1, 6),
)

NANODET_REPVGG_CONFIG = ModelConfig(
    name="nanodet_repvgg",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/nanodet_repvgg",
    class_name="NanoDet",
    model_filename="NanoDet_RepVGG.dxnn",
    ort_on_output_shapes=[(1, 8400, 112)],
    postprocess_result_shape=(1, 6),
)

NANODET_REPVGGA1_CONFIG = ModelConfig(
    name="nanodet_repvgga1",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/nanodet_repvgga1",
    class_name="NanoDet",
    model_filename="NanoDet_RepVGGA1.dxnn",
    ort_on_output_shapes=[(1, 8400, 112)],
    postprocess_result_shape=(1, 6),
)

DAMOYOLO_CONFIG = ModelConfig(
    name="damoyolos",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/damoyolos",
    class_name="DAMOYOLO",
    model_filename="DAMOYoloS.dxnn",
    # DamoYOLO-S split-head: scores [1,8400,80] + boxes [1,8400,4] (640x640 input)
    ort_on_output_shapes=[(1, 8400, 80), (1, 8400, 4)],
    postprocess_result_shape=(1, 6),
)

SSDMV1_CONFIG = ModelConfig(
    name="ssdmv1",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/ssdmv1",
    class_name="SSD",
    model_filename="SSDMv1.dxnn",
    # SSD MobileNet V1: scores [1,3000,21] + boxes [1,3000,4] (300x300 input, VOC 21 classes)
    ort_on_output_shapes=[(1, 3000, 21), (1, 3000, 4)],
    postprocess_result_shape=(1, 6),
)

YOLO26N_OBB_CONFIG = ModelConfig(
    name="yolo26n_obb",
    task=TaskType.OBJECT_DETECTION,
    base_path="obb_detection/yolo26n_obb",
    class_name="YOLOv26Obb",
    model_filename="yolo26n-obb.dxnn",
    detection_output_size=7,
    ort_on_output_shapes=[(1, 300, 7)],
    postprocess_result_shape=(1, 7),
)

# ---------------------------------------------------------------------------
# Pose Estimation
# ---------------------------------------------------------------------------
YOLOV8M_POSE_CONFIG = ModelConfig(
    name="yolov8m_pose",
    task=TaskType.OBJECT_DETECTION,
    base_path="pose_estimation/yolov8m_pose",
    class_name="YOLOv8Pose",
    model_filename="YoloV8M_Pose.dxnn",
    ort_on_output_shapes=[(1, 57, 8400)],
    has_keypoints=True, num_keypoints=17, keypoint_dim=3,
    detection_output_size=57, postprocess_result_shape=(1, 57),
)

YOLOV8S_POSE_CONFIG = ModelConfig(
    name="yolov8s_pose",
    task=TaskType.OBJECT_DETECTION,
    base_path="pose_estimation/yolov8s_pose",
    class_name="YOLOv8Pose",
    model_filename="YoloV8S_Pose.dxnn",
    ort_on_output_shapes=[(1, 57, 8400)],
    has_keypoints=True, num_keypoints=17, keypoint_dim=3,
    detection_output_size=57, postprocess_result_shape=(1, 57),
)

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
BISENETV1_CONFIG = ModelConfig(
    name="bisenetv1",
    task=TaskType.SEMANTIC_SEGMENTATION,
    base_path="semantic_segmentation/bisenetv1",
    class_name="BiSeNetV1",
    model_filename="BiSeNetV1.dxnn",
    ort_on_output_shapes=[(1, 19, 224, 224)],
    postprocess_result_shape=(224, 224),
)

BISENETV2_CONFIG = ModelConfig(
    name="bisenetv2",
    task=TaskType.SEMANTIC_SEGMENTATION,
    base_path="semantic_segmentation/bisenetv2",
    class_name="BiSeNetV2",
    model_filename="BiSeNetV2.dxnn",
    ort_on_output_shapes=[(1, 19, 224, 224)],
    postprocess_result_shape=(224, 224),
)

SEGFORMER_CONFIG = ModelConfig(
    name="segformer_b0_512x1024",
    task=TaskType.SEMANTIC_SEGMENTATION,
    base_path="semantic_segmentation/segformer_b0_512x1024",
    class_name="SegFormer",
    model_filename="SegFormer_B0_512x1024.dxnn",
    ort_on_output_shapes=[(1, 19, 128, 256)],
    postprocess_result_shape=(512, 1024),
)

YOLOV5_SEG_CONFIG = ModelConfig(
    name="yolov5s_seg",
    task=TaskType.INSTANCE_SEGMENTATION,
    base_path="instance_segmentation/yolov5s_seg",
    class_name="YOLOv5Seg",
    model_filename="YoloV5S_Seg.dxnn",
    ort_on_output_shapes=[(1, 116, 8400), (1, 32, 160, 160)],
    postprocess_result_shape=[(1, 6), (1, 640, 640)],
)

YOLOV8S_SEG_CONFIG = ModelConfig(
    name="yolov8s_seg",
    task=TaskType.INSTANCE_SEGMENTATION,
    base_path="instance_segmentation/yolov8s_seg",
    class_name="YOLOv8Seg",
    model_filename="YoloV8S_Seg.dxnn",
    ort_on_output_shapes=[(1, 116, 8400), (1, 32, 160, 160)],
    postprocess_result_shape=[(1, 6), (1, 640, 640)],
)

# ---------------------------------------------------------------------------
# Depth Estimation
# ---------------------------------------------------------------------------
FASTDEPTH_CONFIG = ModelConfig(
    name="fastdepth",
    task=TaskType.DEPTH_ESTIMATION,
    base_path="depth_estimation/fastdepth_1",
    class_name="FastDepth",
    model_filename="FastDepth.dxnn",
    ort_on_output_shapes=[(1, 1, 224, 224)],
    postprocess_result_shape=(224, 224),
)

FASTDEPTH_1_CONFIG = ModelConfig(
    name="fastdepth_1",
    task=TaskType.DEPTH_ESTIMATION,
    base_path="depth_estimation/fastdepth_1",
    class_name="FastDepth",
    model_filename="FastDepth_1.dxnn",
    ort_on_output_shapes=[(1, 1, 224, 224)],
    postprocess_result_shape=(224, 224),
)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
ARCFACE_CONFIG = ModelConfig(
    name="arcface_mobilefacenet",
    task=TaskType.CLASSIFICATION,
    base_path="embedding/arcface_mobilefacenet",
    class_name="ArcFace",
    model_filename="ArcFace_MobileFaceNet.dxnn",
    ort_on_output_shapes=[(1, 128)],
)

OSNET_CONFIG = ModelConfig(
    name="osnet0_25",
    task=TaskType.CLASSIFICATION,
    base_path="classification/osnet0_25",
    class_name="OSNet",
    model_filename="OSNet0_25.dxnn",
    ort_on_output_shapes=[(1, 512)],
)

# ---------------------------------------------------------------------------
# Image Restoration / Enhancement / Super-Resolution
# ---------------------------------------------------------------------------
DNCNN_15_CONFIG = ModelConfig(
    name="dncnn_15",
    task=TaskType.IMAGE_RESTORATION,
    base_path="image_denoising/dncnn_15",
    class_name="DnCNN",
    model_filename="DnCNN_15.dxnn",
    ort_on_output_shapes=[(1, 1, 50, 50)],
    postprocess_result_shape=(50, 50),
)

ZERO_DCE_CONFIG = ModelConfig(
    name="zero_dce",
    task=TaskType.IMAGE_ENHANCEMENT,
    base_path="image_enhancement/zero_dce",
    class_name="ZeroDCE",
    model_filename="ZeroDCE.dxnn",
    ort_on_output_shapes=[(1, 3, 224, 224)],
    postprocess_result_shape=(224, 224),
)

ESPCN_X4_CONFIG = ModelConfig(
    name="espcn_x4",
    task=TaskType.SUPER_RESOLUTION,
    base_path="super_resolution/espcn_x4",
    class_name="ESPCN",
    model_filename="ESPCN_x4.dxnn",
    ort_on_output_shapes=[(1, 3, 448, 448)],
    postprocess_result_shape=(448, 448),
)

# ---------------------------------------------------------------------------
# Hand Landmark
# ---------------------------------------------------------------------------
HAND_LANDMARK_CONFIG = ModelConfig(
    name="hand_landmark_lite",
    task=TaskType.OBJECT_DETECTION,
    base_path="hand_landmark/handlandmarklite_1",
    class_name="HandLandmark",
    model_filename="HandLandmarkLite.dxnn",
    # Hand Landmark: 21 keypoints × 3 coords (x, y, z)
    ort_on_output_shapes=[(1, 63)],
    detection_output_size=6,
    postprocess_result_shape=(1, 6),
)

# ---------------------------------------------------------------------------
# Classification — additional representatives
# ---------------------------------------------------------------------------
ALEXNET_CONFIG = ModelConfig(
    name="alexnet",
    task=TaskType.CLASSIFICATION,
    base_path="classification/alexnet",
    class_name="AlexNet",
    model_filename="AlexNet.dxnn",
    ort_on_output_shapes=[(1, 1000)],
)

MOBILENETV2_CONFIG = ModelConfig(
    name="mobilenetv2",
    task=TaskType.CLASSIFICATION,
    base_path="classification/mobilenetv2",
    class_name="MobileNetV2",
    model_filename="MobileNetV2.dxnn",
    ort_on_output_shapes=[(1, 1000)],
)

RESNET50_CONFIG = ModelConfig(
    name="resnet50",
    task=TaskType.CLASSIFICATION,
    base_path="classification/resnet50",
    class_name="ResNet50",
    model_filename="ResNet50.dxnn",
    ort_on_output_shapes=[(1, 1000)],
)

MODEL_CONFIGS = {
    # Classification
    "efficientnet": EFFICIENTNET_CONFIG,
    "alexnet": ALEXNET_CONFIG,
    "mobilenetv2": MOBILENETV2_CONFIG,
    "resnet50": RESNET50_CONFIG,
    # Face Detection
    "scrfd": SCRFD_CONFIG,
    "scrfd2_5g": SCRFD2_5G_CONFIG,
    "scrfd10g": SCRFD10G_CONFIG,
    "yolov5face": YOLOV5FACE_CONFIG,
    "yolov5m_face": YOLOV5M_FACE_CONFIG,
    "yolov7_face": YOLOV7_FACE_CONFIG,
    "yolov7s_face": YOLOV7S_FACE_CONFIG,
    "yolov7_w6_face": YOLOV7_W6_FACE_CONFIG,
    "yolov7_w6_tta_face": YOLOV7_W6_TTA_FACE_CONFIG,
    "retinaface": RETINAFACE_CONFIG,
    # Object Detection
    "yolov5": YOLOV5_CONFIG,
    "yolov5n": YOLOV5N_CONFIG,
    "yolov5m": YOLOV5M_CONFIG,
    "yolov5l": YOLOV5L_CONFIG,
    "yolov5_ppu": YOLOV5_PPU_CONFIG,
    "yolov5s_ppu": YOLOV5S_PPU_CONFIG,
    "yolov7": YOLOV7_CONFIG,
    "yolov7tiny": YOLOV7TINY_CONFIG,
    "yolov7_ppu": YOLOV7_PPU_CONFIG,
    "yolov8": YOLOV8_CONFIG,
    "yolov8s": YOLOV8S_CONFIG,
    "yolov9": YOLOV9_CONFIG,
    "yolov9c": YOLOV9C_CONFIG,
    "yolov10": YOLOV10_CONFIG,
    "yolov11": YOLOV11_CONFIG,
    "yolov12": YOLOV12_CONFIG,
    "yolov26": YOLOV26_CONFIG,
    "yolo26n": YOLO26N_CONFIG,
    "yolov26obb": YOLOV26OBB_CONFIG,
    "yolo26n_obb": YOLO26N_OBB_CONFIG,
    "yolox": YOLOX_CONFIG,
    "yolox_leaky": YOLOX_LEAKY_CONFIG,
    "nanodet_repvgg": NANODET_REPVGG_CONFIG,
    "nanodet_repvgga1": NANODET_REPVGGA1_CONFIG,
    "damoyolo": DAMOYOLO_CONFIG,
    "ssdmv1": SSDMV1_CONFIG,
    # Pose Estimation
    "yolov5pose": YOLOV5POSE_CONFIG,
    "yolov5pose_ppu": YOLOV5POSE_PPU_CONFIG,
    "yolov8m_pose": YOLOV8M_POSE_CONFIG,
    "yolov8s_pose": YOLOV8S_POSE_CONFIG,
    # Segmentation
    "yolov8seg": YOLOV8SEG_CONFIG,
    "yolov5s_seg": YOLOV5_SEG_CONFIG,
    "yolov8s_seg": YOLOV8S_SEG_CONFIG,
    "deeplabv3": DEEPLABV3_CONFIG,
    "bisenetv1": BISENETV1_CONFIG,
    "bisenetv2": BISENETV2_CONFIG,
    "segformer": SEGFORMER_CONFIG,
    # Depth Estimation
    "fastdepth": FASTDEPTH_CONFIG,
    "fastdepth_1": FASTDEPTH_1_CONFIG,
    # Embedding
    "arcface": ARCFACE_CONFIG,
    "osnet": OSNET_CONFIG,
    # Image Restoration
    "dncnn_15": DNCNN_15_CONFIG,
    "zero_dce": ZERO_DCE_CONFIG,
    "espcn_x4": ESPCN_X4_CONFIG,
    # Hand Landmark
    "hand_landmark": HAND_LANDMARK_CONFIG,
}
