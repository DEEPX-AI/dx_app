"""
Hand Landmark Lite Factory
"""

from common.base import IHandLandmarkFactory
from common.processors import SimpleResizePreprocessor, HandLandmarkPostprocessor
from common.visualizers import HandLandmarkVisualizer


class Handlandmarklite_1Factory(IHandLandmarkFactory):
    """Factory for creating HandLandmarkLite components."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def create_preprocessor(self, input_width: int, input_height: int):
        return SimpleResizePreprocessor(input_width, input_height)
    
    def create_postprocessor(self, input_width: int, input_height: int):
        return HandLandmarkPostprocessor(input_width, input_height, self.config)
    
    def create_visualizer(self):
        return HandLandmarkVisualizer()
    
    def get_model_name(self) -> str:
        return "handlandmarklite_1"
    
    def get_task_type(self) -> str:
        return "hand_landmark"
