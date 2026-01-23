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
    name="efficientnet",
    task=TaskType.CLASSIFICATION,
    base_path="classification/efficientnet",
    class_name="EfficientNet",
    model_filename="EfficientNetB0_8.dxnn",
    ort_on_output_shapes=[(1, 1000)],
    ort_off_output_shapes=[(1, 1000)],
)

SCRFD_CONFIG = ModelConfig(
    name="scrfd",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/scrfd",
    class_name="SCRFD",
    model_filename="SCRFD500M_1.dxnn",
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

SCRFD_PPU_CONFIG = ModelConfig(
    name="scrfd",
    task=TaskType.OBJECT_DETECTION,
    base_path="ppu/scrfd_ppu",
    class_name="SCRFD_PPU",
    model_filename="SCRFD500M_PPU.dxnn",
    ort_on_output_shapes=[(1, 1, 32)],
    has_keypoints=True,
    num_keypoints=5,
    keypoint_dim=2,
    detection_output_size=16,
    postprocess_result_shape=(1, 16),
)

YOLOX_CONFIG = ModelConfig(
    name="yolox",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolox",
    class_name="YOLOX",
    model_filename="YOLOX-S_1.dxnn",
    ort_on_output_shapes=[(1, 5376, 85)],
    postprocess_result_shape=(1, 6),
)

YOLOV5FACE_CONFIG = ModelConfig(
    name="yolov5face",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov5face",
    class_name="YOLOv5Face",
    model_filename="YOLOV5S_Face-1.dxnn",
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
    base_path="object_detection/yolov5pose",
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
    model_filename="YOLOV5Pose_PPU.dxnn",
    ort_on_output_shapes=[(1, 1, 256)],
    has_keypoints=True,
    num_keypoints=17,
    keypoint_dim=3,
    detection_output_size=57,
    postprocess_result_shape=(1, 57),
)

YOLOV5_CONFIG = ModelConfig(
    name="yolov5",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov5",
    class_name="YOLOv5",
    model_filename="YOLOV5S_6.dxnn",
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
    base_path="ppu/yolov5_ppu",
    class_name="YOLOv5_PPU",
    model_filename="YOLOV5S_PPU.dxnn",
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
    name="yolov8",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov8",
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
    name="yolov8seg",
    task=TaskType.INSTANCE_SEGMENTATION,
    base_path="instance_segmentation/yolov8seg",
    class_name="YOLOv8Seg",
    model_filename="YOLOV8N_SEG-1.dxnn",
    ort_on_output_shapes=[(1, 116, 8400),
                          (1, 32, 160, 160)],
    postprocess_result_shape=[(1, 6),
                              (1, 640, 640)]
)

YOLOV9_CONFIG = ModelConfig(
    name="yolov9",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov9",
    class_name="YOLOv9",
    model_filename="YOLOV9S.dxnn",
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
    name="yolov10",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov10",
    class_name="YOLOv10",
    model_filename="YOLOV10N-1.dxnn",
    ort_on_output_shapes=[(1, 300, 6)],
    postprocess_result_shape=(1, 6),
)

YOLOV11_CONFIG = ModelConfig(
    name="yolov11",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov11",
    class_name="YOLOv11",
    model_filename="YOLOV11N.dxnn",
    ort_on_output_shapes=[(1, 84, 8400)],
    postprocess_result_shape=(1, 6),
)

YOLOV12_CONFIG = ModelConfig(
    name="yolov12",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov12",
    class_name="YOLOv12",
    model_filename="YOLOV12N-1.dxnn",
    ort_on_output_shapes=[(1, 84, 8400)],
    postprocess_result_shape=(1, 6),
)

YOLOV26_CONFIG = ModelConfig(
    name="yolov26",
    task=TaskType.OBJECT_DETECTION,
    base_path="object_detection/yolov26",
    class_name="YOLOv26",
    model_filename="yolo26s-1.dxnn",
    ort_on_output_shapes=[(1, 300, 6)],
    postprocess_result_shape=(1, 6),
)

DEEPLABV3_CONFIG = ModelConfig(
    name="deeplabv3",
    task=TaskType.SEMANTIC_SEGMENTATION,
    base_path="semantic_segmentation/deeplabv3",
    class_name="DeepLabv3",
    model_filename="DeepLabV3PlusMobileNetV2_2.dxnn",
    ort_on_output_shapes=[(1, 19, 640, 640)],
    ort_off_output_shapes=[(1, 19, 640, 640)],
    postprocess_result_shape=(640, 640),
)

MODEL_CONFIGS = {
    "efficientnet": EFFICIENTNET_CONFIG,
    "scrfd": SCRFD_CONFIG,
    "scrfd_ppu": SCRFD_PPU_CONFIG,
    "yolov5": YOLOV5_CONFIG,
    "yolov5_ppu": YOLOV5_PPU_CONFIG,
    "yolov5face": YOLOV5FACE_CONFIG,
    "yolov5pose": YOLOV5POSE_CONFIG,
    "yolov5pose_ppu": YOLOV5POSE_PPU_CONFIG,
    "yolov7": YOLOV7_CONFIG,
    "yolov7_ppu": YOLOV7_PPU_CONFIG,
    "yolov8": YOLOV8_CONFIG,
    "yolov8seg": YOLOV8SEG_CONFIG,
    "yolov9": YOLOV9_CONFIG,
    "yolox": YOLOX_CONFIG,
    "deeplabv3": DEEPLABV3_CONFIG,
}
