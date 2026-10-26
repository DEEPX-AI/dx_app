"""
Base interfaces for DX-APP
"""

from .i_input_source import IInputSource, InputType
from .i_processor import IPreprocessor, IPostprocessor, PreprocessContext
from .i_processor import DetectionResult, SegmentationResult, ClassificationResult
from .i_processor import Keypoint, PoseResult, InstanceSegResult, OBBResult
from .i_processor import EmbeddingResult, SuperResolutionResult, EnhancedImageResult, FaceAlignmentResult
from .i_processor import HandLandmarkResult, AnomalyResult
from .i_processor import GalleryMatch, RetrievalResult
from .i_visualizer import IVisualizer
from .i_factory import IDetectionFactory, ISegmentationFactory, IClassificationFactory
from .i_factory import IPoseFactory, IInstanceSegFactory, IFaceFactory, IOBBFactory
from .i_factory import IDepthEstimationFactory, IRestorationFactory
from .i_factory import IEmbeddingFactory, IFaceAlignmentFactory, IHandLandmarkFactory
from .i_factory import IAnomalyDetectionFactory

__all__ = [
    "AnomalyResult",
    "GalleryMatch", "RetrievalResult",
    "IAnomalyDetectionFactory",
    'IInputSource', 'InputType',
    'IPreprocessor', 'IPostprocessor', 'PreprocessContext',
    'DetectionResult', 'SegmentationResult', 'ClassificationResult',
    'Keypoint', 'PoseResult', 'InstanceSegResult', 'OBBResult',
    'EmbeddingResult', 'SuperResolutionResult', 'EnhancedImageResult', 'FaceAlignmentResult',
    'HandLandmarkResult',
    'IVisualizer',
    'IDetectionFactory', 'ISegmentationFactory', 'IClassificationFactory',
    'IPoseFactory', 'IInstanceSegFactory', 'IFaceFactory', 'IOBBFactory',
    'IDepthEstimationFactory', 'IRestorationFactory',
    'IEmbeddingFactory', 'IFaceAlignmentFactory', 'IHandLandmarkFactory',
]
