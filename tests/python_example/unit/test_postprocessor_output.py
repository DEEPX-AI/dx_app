"""Level 2: Postprocessor output structure tests.

Feed dummy tensors and verify output types and shapes.
Tests only postprocessors with simple, well-defined tensor formats.
Complex multi-scale models (YOLO detection, SCRFD) are excluded
as their tensor formats are tightly coupled to model architecture.
"""
import pytest
import numpy as np

from common.processors.classification_postprocessor import ClassificationPostprocessor
from common.processors.depth_postprocessor import DepthEstimationPostprocessor
from common.processors.embedding_postprocessor import GenericEmbeddingPostprocessor
from common.processors.segmentation_postprocessor import SemanticSegmentationPostprocessor


# ── Classification ────────────────────────────────────────────────

class TestClassificationOutput:
    """ClassificationPostprocessor with [1, num_classes] logit tensor."""

    def test_basic_output(self, ctx):
        pp = ClassificationPostprocessor(224, 224)
        logits = np.random.randn(1, 1000).astype(np.float32)
        results = pp.process([logits], ctx)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_top_k_count(self, ctx):
        pp = ClassificationPostprocessor(224, 224, config={"top_k": 3})
        logits = np.random.randn(1, 1000).astype(np.float32)
        results = pp.process([logits], ctx)
        assert len(results) <= 3

    def test_result_fields(self, ctx):
        pp = ClassificationPostprocessor(224, 224, config={"top_k": 1})
        logits = np.zeros((1, 10), dtype=np.float32)
        logits[0, 7] = 10.0  # class 7 should have highest confidence
        results = pp.process([logits], ctx)
        assert len(results) >= 1
        top = results[0]
        assert hasattr(top, 'class_id')
        assert hasattr(top, 'confidence')
        assert top.class_id == 7
        assert top.confidence > 0.0

    def test_confidences_sum_to_one(self, ctx):
        """Softmax outputs should approximately sum to 1 over all classes."""
        pp = ClassificationPostprocessor(224, 224, config={"top_k": 1000})
        logits = np.random.randn(1, 100).astype(np.float32)
        results = pp.process([logits], ctx)
        total_conf = sum(r.confidence for r in results)
        assert 0.99 < total_conf < 1.01

    def test_flat_tensor(self, ctx):
        """Should handle 1D tensor [num_classes]."""
        pp = ClassificationPostprocessor(224, 224)
        logits = np.random.randn(1000).astype(np.float32)
        results = pp.process([logits], ctx)
        assert len(results) > 0


# ── Depth Estimation ──────────────────────────────────────────────

class TestDepthOutput:
    """DepthEstimationPostprocessor with [1, 1, H, W] tensor."""

    def test_basic_output(self, ctx):
        pp = DepthEstimationPostprocessor(256, 256)
        depth = np.random.rand(1, 1, 256, 256).astype(np.float32)
        results = pp.process([depth], ctx)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_depth_map_shape(self, ctx):
        pp = DepthEstimationPostprocessor(256, 256)
        depth = np.random.rand(1, 1, 256, 256).astype(np.float32)
        results = pp.process([depth], ctx)
        result = results[0]
        assert hasattr(result, 'depth_map') or hasattr(result, 'mask')


# ── Embedding ─────────────────────────────────────────────────────

class TestEmbeddingOutput:
    """GenericEmbeddingPostprocessor with [1, D] feature vector."""

    def test_basic_output(self, ctx):
        pp = GenericEmbeddingPostprocessor(112, 112)
        features = np.random.randn(1, 512).astype(np.float32)
        results = pp.process([features], ctx)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_embedding_is_normalized(self, ctx):
        """Output embedding should be L2-normalized."""
        pp = GenericEmbeddingPostprocessor(112, 112)
        features = np.random.randn(1, 512).astype(np.float32) * 100
        results = pp.process([features], ctx)
        result = results[0]
        assert hasattr(result, 'embedding')
        norm = np.linalg.norm(result.embedding)
        assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm}"

    def test_flat_input(self, ctx):
        """Should handle 1D input [D]."""
        pp = GenericEmbeddingPostprocessor(112, 112)
        features = np.random.randn(512).astype(np.float32)
        results = pp.process([features], ctx)
        assert len(results) >= 1


# ── Semantic Segmentation ─────────────────────────────────────────

class TestSegmentationOutput:
    """SemanticSegmentationPostprocessor with [1, C, H, W] logits."""

    def test_basic_output(self, ctx):
        pp = SemanticSegmentationPostprocessor(512, 512, config={"num_classes": 19})
        logits = np.random.randn(1, 19, 128, 128).astype(np.float32)
        results = pp.process([logits], ctx)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_mask_shape(self, ctx):
        pp = SemanticSegmentationPostprocessor(512, 512, config={"num_classes": 19})
        logits = np.random.randn(1, 19, 128, 128).astype(np.float32)
        results = pp.process([logits], ctx)
        result = results[0]
        assert hasattr(result, 'mask')
        assert result.mask.ndim == 2  # [H, W]

    def test_mask_values_in_range(self, ctx):
        """Mask values should be valid class indices."""
        n_classes = 19
        pp = SemanticSegmentationPostprocessor(512, 512, config={"num_classes": n_classes})
        logits = np.random.randn(1, n_classes, 64, 64).astype(np.float32)
        results = pp.process([logits], ctx)
        mask = results[0].mask
        assert mask.min() >= 0
        assert mask.max() < n_classes

    def test_preargmaxed_input(self, ctx):
        """Should handle pre-argmaxed [H, W] integer input."""
        pp = SemanticSegmentationPostprocessor(512, 512)
        mask = np.random.randint(0, 19, (128, 128)).astype(np.int32)
        results = pp.process([mask], ctx)
        assert len(results) >= 1
