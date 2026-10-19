# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""EfficientAD's published anomaly map, computed from all three of its networks.

MEASURED on DX-RT 3.5.0, the EfficientAD-M triple at 256x256::

    efficientad-m-teacher_256x256      out teacher_features [1, 384, 64, 64]
    efficientad-m-student_256x256      out student_features [1, 768, 64, 64]
    efficientad-m-autoencoder_256x256  out teacher_features [1, 384, 64, 64]

768 = 384 + 384. The student predicts the TEACHER with its first half and the
AUTOENCODER with its second, and EfficientAD scores the two disagreements::

    st = mean((teacher     - student[:384])**2, axis=C)
    ae = mean((autoencoder - student[384:])**2, axis=C)
    anomaly = 0.5*norm(st) + 0.5*norm(ae)

This is why a one-network example could not be right: the magnitude of a single
feature map is not a disagreement, and a disagreement is the entire method.

**What is still missing, and why.** The reference implementation normalises with
q_st/q_ae quantiles fitted on the training set, which is what makes its score
comparable across images and against a published MVTec number. A `.dxnn` carries no
such constants, so each branch is normalised against its own frame here. The
structure -- which pixels disagree, and how strongly relative to the rest of the
frame -- is the real thing; the absolute number is frame-relative and says so.

Teacher and autoencoder are both (1, 384, H, W), so nothing in the data distinguishes
them. Which output is which is therefore declared, not inferred: each variant config
carries ``config.roles`` naming its outputs in arrival order, because the PRIMARY
model is whichever one ``-m`` names and the companions follow it. Without that, a run
started from the autoencoder variant would silently swap the two 384-channel maps and
compute a different quantity under the same name.

Absent ``roles`` the class falls back to shape order -- the 768-channel map is the
student, the remaining two are teacher then autoencoder as they arrive.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..base import AnomalyResult, IPostprocessor, PreprocessContext


def _shapes(outputs: Sequence[np.ndarray]) -> str:
    return ", ".join("(" + ", ".join(str(d) for d in o.shape) + ")" for o in outputs)


def _to_chw(out: np.ndarray) -> np.ndarray:
    """``(C, H, W)`` from either layout.

    The spatial grid of these networks is square (64x64 at a 256x256 input), which is
    what tells the layouts apart: in NHWC the equal pair is FIRST, in NCHW it is LAST.
    A wrong guess here transposes the map without erroring.
    """
    arr = out[0] if out.ndim == 4 else out
    if arr.ndim != 3:
        raise ValueError(
            "[DXAPP] [ERROR] EfficientADPostprocessor - expected a (1, C, H, W) "
            f"feature map, got shape {out.shape}."
        )
    a, b, c = arr.shape
    if a == b and b != c:
        return np.transpose(arr, (2, 0, 1))      # NHWC
    return arr                                    # NCHW


def _normalise(response: np.ndarray) -> np.ndarray:
    """Frame-relative 0..1. A flat branch stays at 0 rather than becoming all-hot."""
    low, high = float(response.min()), float(response.max())
    span = high - low
    return ((response - low) / span) if span > 1e-12 else np.zeros_like(response)


class EfficientADPostprocessor(IPostprocessor):
    """Student-teacher + autoencoder disagreement -> per-pixel anomaly map."""

    def __init__(self, input_width: int = 256, input_height: int = 256,
                 config: dict = None):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        # The percentile reported as `score`; 99 rather than max, so one hot pixel of
        # quantisation noise cannot dominate a frame's severity.
        self.score_percentile = float(self.config.get("score_percentile", 99.0))
        self.st_weight = float(self.config.get("st_weight", 0.5))
        self.ae_weight = float(self.config.get("ae_weight", 0.5))
        # Output roles in arrival order, e.g. ["student", "teacher", "autoencoder"].
        self.roles = [str(r) for r in (self.config.get("roles") or [])]

    # --------------------------------------------------------------- unpacking
    def _split(self, outputs: Sequence[np.ndarray]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) != 3:
            raise ValueError(
                "[DXAPP] [ERROR] EfficientADPostprocessor - needs all three "
                "EfficientAD networks, got "
                f"{len(outputs)} output(s): {_shapes(outputs)}\n"
                "  Expected a 768-channel student plus a 384-channel teacher and a "
                "384-channel autoencoder, in that order.\n"
                "  The student alone cannot be scored: 768 = 384 (student-teacher "
                "branch) + 384 (autoencoder branch), and both halves need their "
                "target to be a disagreement at all."
            )
        maps = [_to_chw(o).astype(np.float32) for o in outputs]

        if len(self.roles) == len(maps):
            by_role = dict(zip(self.roles, maps))
            missing = {"student", "teacher", "autoencoder"} - set(by_role)
            if missing:
                raise ValueError(
                    "[DXAPP] [ERROR] EfficientADPostprocessor - config.roles does not "
                    f"name every network: missing {sorted(missing)}, got {self.roles}."
                )
            student = by_role["student"]
            teacher = by_role["teacher"]
            autoencoder = by_role["autoencoder"]
        else:
            student_idx = int(np.argmax([m.shape[0] for m in maps]))
            student = maps[student_idx]
            teacher, autoencoder = [m for i, m in enumerate(maps) if i != student_idx]

        if student.shape[0] != teacher.shape[0] + autoencoder.shape[0] \
                or teacher.shape[0] != autoencoder.shape[0]:
            raise ValueError(
                "[DXAPP] [ERROR] EfficientADPostprocessor - the student's channels do "
                "not split into its two targets.\n"
                f"  student {student.shape[0]}, teacher {teacher.shape[0]}, "
                f"autoencoder {autoencoder.shape[0]}\n"
                "  EfficientAD's student emits exactly twice its teacher's channels."
            )
        return student, teacher, autoencoder

    # ---------------------------------------------------------------- decoding
    def process(self, outputs: List[np.ndarray],
                ctx: PreprocessContext) -> List[AnomalyResult]:
        student, teacher, autoencoder = self._split(outputs)
        half = teacher.shape[0]

        st = np.mean((teacher - student[:half]) ** 2, axis=0)
        ae = np.mean((autoencoder - student[half:]) ** 2, axis=0)

        # Two maps, two jobs. The HEATMAP is normalised against its own frame, which
        # is what makes a hot region visible whatever the absolute magnitudes are.
        # The SCORE is read off the RAW disagreement, because a per-frame normalised
        # map has roughly the same 99th percentile whether the anomaly is faint or
        # glaring -- it could not rank one frame against another, which is the only
        # thing a single number is good for here.
        combined = self.st_weight * _normalise(st) + self.ae_weight * _normalise(ae)
        raw = self.st_weight * st + self.ae_weight * ae

        heatmap = cv2.resize(combined, (ctx.original_width, ctx.original_height),
                             interpolation=cv2.INTER_LINEAR)
        heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

        return [AnomalyResult(
            heatmap=heatmap,
            # Comparable across frames for THIS model set, and to nothing else: the
            # published EfficientAD score divides by q_st/q_ae quantiles fitted on the
            # training set, and a .dxnn carries no such constants.
            score=float(np.percentile(raw, self.score_percentile)),
            channels=int(student.shape[0]),
        )]

    def get_model_name(self) -> str:
        return "efficientad"
