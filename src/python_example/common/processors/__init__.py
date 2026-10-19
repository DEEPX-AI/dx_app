"""
Common Processors - Reusable preprocessors and postprocessors
DX-APP
"""

from .letterbox_preprocessor import LetterboxPreprocessor
from .simple_resize_preprocessor import SimpleResizePreprocessor
from .grayscale_preprocessor import GrayscaleResizePreprocessor
from .yolo_postprocessor import (
    YOLOv5Postprocessor,
    YOLOv8Postprocessor,
    YOLOXPostprocessor,
)
from .fast_yolo_postprocessor import FastYOLOv5Postprocessor
from .classification_postprocessor import ClassificationPostprocessor
from .segmentation_postprocessor import SemanticSegmentationPostprocessor
from .fast_segmentation_postprocessor import FastSegmentationPostprocessor
from .face_postprocessor import (
    FaceResult,
    SCRFDPostprocessor,
    YOLOv5FacePostprocessor,
    YOLOv7FacePostprocessor,
)
from .pose_postprocessor import (
    YOLOv5PosePostprocessor,
    YOLOv8PosePostprocessor,
)
from .instance_seg_postprocessor import (
    InstanceSegPostprocessor,
    YOLOv8InstanceSegPostprocessor,
    YOLOv5InstanceSegPostprocessor,
)
from .fast_instance_seg_postprocessor import FastInstanceSegPostprocessor
from .preopt_postprocessor import (
    PreoptDetectionPostprocessor,
    PreoptPosePostprocessor,
    PreoptSegPostprocessor,
)
from .rtdetr_postprocessor import (
    MaskRTDETRPostprocessor,
    RTDETRPostprocessor,
)
# The base the arcface/clip wrappers already subclass. Exported so a variant
# whose embedding is neither a face nor a CLIP image -- PP-ShiTu retrieval,
# RepVGG person ReID -- can report its own model_type/model_name instead of
# borrowing ArcFacePostprocessor's hardcoded "face_embedding"/"arcface".
from .embedding_postprocessor import GenericEmbeddingPostprocessor
from .matting_postprocessor import PPMattingPostprocessor
from .gallery_retrieval_postprocessor import GalleryRetrievalPostprocessor
from .efficientad_postprocessor import EfficientADPostprocessor
from .anomaly_postprocessor import AnomalyFeaturePostprocessor
from .picodet_postprocessor import PicoDetPostprocessor
from .damoyolo_postprocessor import DamoYoloPostprocessor
from .ssd_postprocessor import SSDPostprocessor
from .tflite_det_postprocessor import TFLiteDetectionPostprocessor
from .nanodet_postprocessor import NanoDetPostprocessor
from .depth_postprocessor import DepthEstimationPostprocessor, DepthResult
from .fast_depth_postprocessor import FastDepthEstimationPostprocessor
from .restoration_postprocessor import DnCNNPostprocessor, RestorationResult, RealESRGANPostprocessor
from .obb_postprocessor import OBBPostprocessor
from .ppu_postprocessor import (
    PPUPostprocessor,
    YOLOv5PPUPostprocessor,
    YOLOv7PPUPostprocessor,
    YOLOv8PPUPostprocessor,
    YOLOv10PPUPostprocessor,
    YOLOXPPUPostprocessor,
    SCRFDPPUPostprocessor,
    YOLOv5PosePPUPostprocessor,
)
from .centernet_postprocessor import CenterNetPostprocessor
from .efficientdet_postprocessor import EfficientDetPostprocessor
from .fast_efficientdet_postprocessor import FastEfficientDetPostprocessor
from .retinaface_postprocessor import RetinaFacePostprocessor
from .mediapipe_hand_postprocessor import MediaPipeHandPostprocessor
from .ulfg_postprocessor import ULFGPostprocessor
from .centerpose_postprocessor import CenterPosePostprocessor
from .yolact_postprocessor import YOLACTPostprocessor
from .fast_yolact_postprocessor import FastYOLACTPostprocessor
from .espcn_postprocessor import ESPCNPostprocessor
from .zero_dce_postprocessor import ZeroDCEPostprocessor
from .clip_postprocessor import CLIPImagePostprocessor, CLIPTextPostprocessor
from .clip_zeroshot_postprocessor import CLIPZeroShotPostprocessor
from .arcface_postprocessor import ArcFacePostprocessor
from .segformer_postprocessor import SegFormerPostprocessor
from .palm_postprocessor import PalmDetectionPostprocessor
from .tddfa_postprocessor import TDDFAPostprocessor
from .hand_landmark_postprocessor import HandLandmarkPostprocessor
from .attribute_postprocessor import AttributePostprocessor
from .cpp_compat import EmbeddingPostProcess, ZeroDCEPostProcess, PythonFallbackPostProcess
from .vitpose_postprocessor import VitPosePostprocessor
from .superpoint_postprocessor import SuperPointPostprocessor
from .dope_postprocessor import DOPEPostprocessor
from .yolopv2_postprocessor import YOLOPv2Postprocessor
from .sfa3d_postprocessor import SFA3DPostprocessor, Detection3DResult, SFA3D_CLASSES
from .sfa3d_bev_preprocessor import SFA3DBEVPreprocessor, load_kitti_pointcloud, pointcloud_to_bev

__all__ = [
    # Pre-optimized YOLO models: the top-k row table is produced inside the model.
    "PreoptDetectionPostprocessor",
    "PreoptPosePostprocessor",
    "PreoptSegPostprocessor",
    # RT-DETR family: NMS-free query decoding.
    "RTDETRPostprocessor",
    "MaskRTDETRPostprocessor",
    # PaddleSeg matting: a continuous alpha matte, not a class map.
    "GenericEmbeddingPostprocessor",
    "PPMattingPostprocessor",
    # Image retrieval / visual place recognition / person ReID: one descriptor
    # ranked against a gallery built on the NPU at build time.
    "GalleryRetrievalPostprocessor",
    # Anomaly detection: a feature response, not a published anomaly score.
    "AnomalyFeaturePostprocessor",
    "EfficientADPostprocessor",
    # PP-PicoDet: a GFL head delivered one tensor per pyramid level, unlike
    # NanoDet which concatenates them.
    "PicoDetPostprocessor",
    # Preprocessors
    'LetterboxPreprocessor',
    'SimpleResizePreprocessor',
    'GrayscaleResizePreprocessor',
    # Detection Postprocessors
    'YOLOv5Postprocessor',
    'YOLOv8Postprocessor', 
    'YOLOXPostprocessor',
    'FastYOLOv5Postprocessor',
    # Classification Postprocessors
    'ClassificationPostprocessor',
    # Segmentation Postprocessors
    'SemanticSegmentationPostprocessor',
    'FastSegmentationPostprocessor',
    'FastInstanceSegPostprocessor',
    # Face Detection Postprocessors
    'FaceResult',
    'SCRFDPostprocessor',
    'YOLOv5FacePostprocessor',
    'YOLOv7FacePostprocessor',
    # Pose Estimation Postprocessors
    'YOLOv5PosePostprocessor',
    'YOLOv8PosePostprocessor',
    # Instance Segmentation Postprocessors
    'InstanceSegPostprocessor',
    'YOLOv8InstanceSegPostprocessor',
    'YOLOv5InstanceSegPostprocessor',
    # DamoYOLO Postprocessor
    'DamoYoloPostprocessor',
    # SSD Postprocessor
    'SSDPostprocessor',
    # TFLite Detection Postprocessor
    'TFLiteDetectionPostprocessor',
    # NanoDet Postprocessor
    'NanoDetPostprocessor',
    # Depth Estimation Postprocessor
    'DepthEstimationPostprocessor',
    'FastDepthEstimationPostprocessor',
    'DepthResult',
    # Image Restoration Postprocessor
    'DnCNNPostprocessor',
    'RestorationResult',
    # OBB Postprocessors
    'OBBPostprocessor',
    # PPU Postprocessors
    'PPUPostprocessor',
    'YOLOv5PPUPostprocessor',
    'YOLOv7PPUPostprocessor',
    'YOLOv8PPUPostprocessor',
    'YOLOv10PPUPostprocessor',
    'YOLOXPPUPostprocessor',
    'SCRFDPPUPostprocessor',
    'YOLOv5PosePPUPostprocessor',
    # CenterNet Postprocessor
    'CenterNetPostprocessor',
    # EfficientDet Postprocessor
    'EfficientDetPostprocessor',
    'FastEfficientDetPostprocessor',
    # RetinaFace Postprocessor
    'RetinaFacePostprocessor',
    'MediaPipeHandPostprocessor',
    # ULFG Postprocessor
    'ULFGPostprocessor',
    # CenterPose Postprocessor
    'CenterPosePostprocessor',
    # YOLACT Postprocessor
    'YOLACTPostprocessor',
    'FastYOLACTPostprocessor',
    # ESPCN Super-Resolution Postprocessor
    'ESPCNPostprocessor',
    # Zero-DCE Enhancement Postprocessor
    'ZeroDCEPostprocessor',
    # CLIP Postprocessors
    'CLIPImagePostprocessor',
    'CLIPTextPostprocessor',
    'CLIPZeroShotPostprocessor',
    # ArcFace Postprocessor
    'ArcFacePostprocessor',
    # SegFormer Postprocessor
    'SegFormerPostprocessor',
    # Palm Detection Postprocessor
    'PalmDetectionPostprocessor',
    # 3DDFA Face Alignment Postprocessor
    'TDDFAPostprocessor',
    # Hand Landmark Postprocessor
    'HandLandmarkPostprocessor',
    # Attribute Recognition Postprocessor
    'AttributePostprocessor',
    # C++ Postprocess Compatibility (Python fallbacks for missing dx_postprocess.so classes)
    'EmbeddingPostProcess',
    'ZeroDCEPostProcess',
    'PythonFallbackPostProcess',
    'SFA3DPostprocessor',
    'Detection3DResult',
    'SFA3D_CLASSES',
    'SFA3DBEVPreprocessor',
    'load_kitti_pointcloud',
    'pointcloud_to_bev',
]
