"""
Classification Postprocessor

For classification models like EfficientNet, ResNet, etc.

Same behaviour as the C++ EfficientNetPostprocessor
(src/cpp_example/common/processors/classification_postprocessor.hpp):
stable softmax over the logits, no second softmax when the network already
outputs a distribution, ImageNet class names for 1000 classes, and the ranked
top-k on the first result.
"""

import numpy as np
from typing import List, Any

from ..base import IPostprocessor, PreprocessContext, ClassificationResult
from ..utility.labels import IMAGENET_1000

# Same tolerances as the C++ detail::IsAlreadyDistribution().
_DIST_MIN_VALUE = -1e-4
_DIST_SUM_TOLERANCE = 1e-3


def _is_already_distribution(values: np.ndarray) -> bool:
    """True when *values* already are probabilities (network ends in softmax):
    every value >= -1e-4 and the sum within 1e-3 of 1."""
    if values.size == 0:
        return False
    if np.any(values < _DIST_MIN_VALUE):
        return False
    return abs(float(np.sum(values, dtype=np.float64)) - 1.0) <= _DIST_SUM_TOLERANCE


class ClassificationPostprocessor(IPostprocessor):
    """
    Postprocessor for classification models.
    
    Applies softmax (unless the output already is a distribution) and returns
    the top-k predictions; the first one also carries the ranked top-k list.
    """
    
    def __init__(self, input_width: int = 224, input_height: int = 224, config: dict = None):
        """
        Initialize classification postprocessor.
        
        Args:
            input_width: Model input width
            input_height: Model input height
            config: Optional configuration
        """
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.num_classes = self.config.get('num_classes', 1000)
        self.top_k = self.config.get('top_k', 5)
    
    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[ClassificationResult]:
        """
        Process classification model outputs.
        
        Args:
            outputs: Model outputs
            ctx: Preprocessing context
            
        Returns:
            List of ClassificationResult (top-k predictions)
        """
        output = outputs[0]
        
        # Handle single class output (argmax already applied in model)
        if output.size == 1:
            class_id = int(output.item())
            return [ClassificationResult(
                class_id=class_id,
                confidence=1.0,
                class_name=self._class_name(class_id)
            )]
        
        values = output.flatten() if output.ndim > 1 else output
        if _is_already_distribution(values):
            probabilities = values.astype(np.float64)
        else:
            probabilities = self._softmax(values)
        
        # Get top-k (softmax is monotonic: same ranking as the logits)
        k = min(int(self.top_k), probabilities.size)
        top_indices = np.argsort(-probabilities, kind="stable")[:k]
        top_k = [(int(idx), float(probabilities[idx])) for idx in top_indices]
        
        results = []
        for idx, conf in top_k:
            results.append(ClassificationResult(
                class_id=idx,
                confidence=conf,
                class_name=self._class_name(idx)
            ))
        if results:
            results[0].top_k = top_k
        
        return results
    
    def get_model_name(self) -> str:
        return "classification"
    
    def _class_name(self, class_id: int) -> str:
        if self.num_classes == 1000 and 0 <= class_id < len(IMAGENET_1000):
            return IMAGENET_1000[class_id]
        return ""

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax (the max is subtracted first)."""
        x = np.asarray(x, dtype=np.float64)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
