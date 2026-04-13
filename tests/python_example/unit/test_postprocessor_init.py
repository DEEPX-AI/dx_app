"""Level 1: Postprocessor instantiation tests.

Verify that all major postprocessor classes can be created with
default and custom config values, without requiring model files.
"""
import pytest
import numpy as np

from common.processors.classification_postprocessor import ClassificationPostprocessor
from common.processors.depth_postprocessor import DepthEstimationPostprocessor
from common.processors.embedding_postprocessor import GenericEmbeddingPostprocessor
from common.processors.segmentation_postprocessor import SemanticSegmentationPostprocessor
from common.processors.yolo_postprocessor import YOLOv8Postprocessor, YOLOv5Postprocessor
from common.processors.face_postprocessor import SCRFDPostprocessor
from common.processors.pose_postprocessor import YOLOv8PosePostprocessor
from common.processors.instance_seg_postprocessor import YOLOv8InstanceSegPostprocessor


# ── Instantiation with default config ─────────────────────────────

class TestDefaultInstantiation:
    """All postprocessors should be creatable with minimal arguments."""

    def test_classification(self):
        pp = ClassificationPostprocessor(224, 224)
        assert pp is not None

    def test_depth(self):
        pp = DepthEstimationPostprocessor(256, 256)
        assert pp is not None

    def test_embedding(self):
        pp = GenericEmbeddingPostprocessor(112, 112)
        assert pp is not None

    def test_segmentation(self):
        pp = SemanticSegmentationPostprocessor(512, 512)
        assert pp is not None

    def test_yolov8_detection(self):
        pp = YOLOv8Postprocessor(640, 640)
        assert pp is not None

    def test_yolov5_detection(self):
        pp = YOLOv5Postprocessor(640, 640)
        assert pp is not None

    def test_scrfd_face(self):
        pp = SCRFDPostprocessor(640, 640)
        assert pp is not None

    def test_yolov8_pose(self):
        pp = YOLOv8PosePostprocessor(640, 640)
        assert pp is not None

    def test_yolov8_seg(self):
        pp = YOLOv8InstanceSegPostprocessor(640, 640)
        assert pp is not None


# ── Instantiation with custom config ──────────────────────────────

class TestCustomConfig:
    """Postprocessors should accept config dict overrides."""

    def test_classification_custom_topk(self):
        pp = ClassificationPostprocessor(224, 224, config={"top_k": 10})
        assert pp is not None

    def test_detection_custom_thresholds(self):
        pp = YOLOv8Postprocessor(640, 640, config={
            "conf_threshold": 0.5,
            "nms_threshold": 0.3,
        })
        assert pp is not None

    def test_detection_custom_classes(self):
        pp = YOLOv5Postprocessor(640, 640, config={
            "num_classes": 20,
            "conf_threshold": 0.25,
        })
        assert pp is not None

    def test_segmentation_custom(self):
        pp = SemanticSegmentationPostprocessor(512, 512, config={
            "num_classes": 21,
        })
        assert pp is not None

    def test_empty_config(self):
        """Empty config should use defaults."""
        pp = YOLOv8Postprocessor(640, 640, config={})
        assert pp is not None


# ── Edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    """Postprocessors should handle edge-case inputs gracefully."""

    def test_none_config(self):
        pp = ClassificationPostprocessor(224, 224, config=None)
        assert pp is not None

    def test_small_dimensions(self):
        pp = ClassificationPostprocessor(1, 1)
        assert pp is not None

    def test_large_dimensions(self):
        pp = YOLOv8Postprocessor(1920, 1080)
        assert pp is not None
