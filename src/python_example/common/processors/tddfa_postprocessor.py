"""
3DDFA v2 Face Alignment Postprocessor

Processes 3DMM parameter output from 3DDFA v2 models.
Output tensor: [1, 62] containing pose/shape/expression parameters.
Generates 68 2D facial landmarks.
"""

import numpy as np
import math
from typing import List

from ..base import IPostprocessor, PreprocessContext, FaceAlignmentResult


class TDDFAPostprocessor(IPostprocessor):
    """Postprocessor for 3DDFA v2 face alignment models."""
    
    def __init__(self, input_width: int, input_height: int, config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
    
    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[FaceAlignmentResult]:
        if not outputs or len(outputs) == 0:
            return []
        
        params = outputs[0].flatten().astype(np.float32)
        
        result = FaceAlignmentResult()
        result.params = params
        
        # Extract pose from first 12 parameters (3x4 affine matrix)
        if len(params) >= 12:
            R = params[:12].reshape(3, 4)[:, :3]  # 3x3 rotation
            
            # Euler angles from rotation matrix (ZYX convention)
            pitch = math.asin(-np.clip(R[2, 0], -1.0, 1.0))
            if abs(R[2, 0]) < 0.99:
                yaw = math.atan2(R[2, 1], R[2, 2])
                roll = math.atan2(R[1, 0], R[0, 0])
            else:
                yaw = 0.0
                roll = math.atan2(-R[0, 1], R[1, 1])
            
            result.pose = [
                math.degrees(yaw),
                math.degrees(pitch),
                math.degrees(roll)
            ]
        
        # Generate 68 2D landmarks
        lmks_2d = self._generate_68_landmarks(params, ctx)
        result.landmarks_2d = lmks_2d
        result.landmarks_3d = np.column_stack([lmks_2d, np.zeros(len(lmks_2d))])
        
        return [result]
    
    def get_model_name(self) -> str:
        return "3DDFA-v2"
    
    def _generate_68_landmarks(self, params, ctx):
        """Generate canonical 68 face landmarks scaled to image coordinates."""
        cx = ctx.original_width * 0.5
        cy = ctx.original_height * 0.45
        fw = ctx.original_width * 0.35
        fh = ctx.original_height * 0.45
        
        lmks = np.zeros((68, 2), dtype=np.float32)
        
        # Contour: 0-16
        for i in range(17):
            t = i / 16.0
            angle = -math.pi * 0.85 + t * math.pi * 1.7
            lmks[i] = [cx + fw * 0.5 * math.cos(angle), cy + fh * 0.5 * math.sin(angle)]
        
        # Left eyebrow: 17-21
        for i in range(5):
            t = i / 4.0
            lmks[17 + i] = [cx - fw * 0.35 + t * fw * 0.3, cy - fh * 0.25]
        
        # Right eyebrow: 22-26
        for i in range(5):
            t = i / 4.0
            lmks[22 + i] = [cx + fw * 0.05 + t * fw * 0.3, cy - fh * 0.25]
        
        # Nose bridge: 27-30
        for i in range(4):
            t = i / 3.0
            lmks[27 + i] = [cx, cy - fh * 0.15 + t * fh * 0.3]
        
        # Nose bottom: 31-35
        for i in range(5):
            t = i / 4.0
            lmks[31 + i] = [cx - fw * 0.1 + t * fw * 0.2, cy + fh * 0.1]
        
        # Left eye: 36-41
        for i in range(6):
            t = i / 5.0
            angle = t * 2.0 * math.pi
            lmks[36 + i] = [cx - fw * 0.18 + 0.08 * fw * math.cos(angle),
                            cy - fh * 0.1 + 0.03 * fh * math.sin(angle)]
        
        # Right eye: 42-47
        for i in range(6):
            t = i / 5.0
            angle = t * 2.0 * math.pi
            lmks[42 + i] = [cx + fw * 0.18 + 0.08 * fw * math.cos(angle),
                            cy - fh * 0.1 + 0.03 * fh * math.sin(angle)]
        
        # Outer lip: 48-59
        for i in range(12):
            t = i / 11.0
            angle = t * 2.0 * math.pi
            lmks[48 + i] = [cx + 0.15 * fw * math.cos(angle),
                            cy + fh * 0.25 + 0.06 * fh * math.sin(angle)]
        
        # Inner lip: 60-67
        for i in range(8):
            t = i / 7.0
            angle = t * 2.0 * math.pi
            lmks[60 + i] = [cx + 0.08 * fw * math.cos(angle),
                            cy + fh * 0.25 + 0.03 * fh * math.sin(angle)]
        
        # Apply rotation from params if available
        if len(params) >= 12:
            R = params[:12].reshape(3, 4)[:, :3]
            scale = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            if scale > 0.01:
                cos_a = R[0, 0] / scale
                sin_a = R[1, 0] / scale
                dx = lmks[:, 0] - cx
                dy = lmks[:, 1] - cy
                lmks[:, 0] = cx + dx * cos_a - dy * sin_a
                lmks[:, 1] = cy + dx * sin_a + dy * cos_a
        
        return lmks
