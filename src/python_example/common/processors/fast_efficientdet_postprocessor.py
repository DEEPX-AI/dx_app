"""Fast EfficientDet multi-output postprocessor (exact, opt-in).

The standard :class:`EfficientDetPostprocessor` BiFPN path
(``_process_multi_output``) spends most of its time on two per-frame operations
that profiling on real NPU output (EfficientDet-D1: 76,725 anchors x 90 classes)
showed dominate the cost:

* ``np.argmax`` over *every* anchor's class scores -- even though only anchors that
  pass the score threshold ever use their class id, and
* ``np.percentile`` / ``np.median`` over *all* box values to decide whether the box
  tensor holds anchor regressions or absolute coordinates -- recomputed on every
  frame even though a given model always emits the same format.

This variant keeps the per-anchor ``max`` (which is required to gate) and the
downstream :meth:`_decode_anchor_results` / :meth:`_decode_results` unchanged, but

* computes ``argmax`` only for the anchors that pass the score threshold and
  scatters the result into a zero-filled class-id array (non-survivors are masked
  out downstream, so their placeholder id is never read), and
* computes the regression-vs-absolute decision once and caches it.

Both downstream decoders re-apply ``scores >= score_threshold`` and use only the
surviving anchors, so the produced detections are byte-identical to the standard
path (verified by the parity unit tests). It is gated behind ``--fast-postprocess``
so the default path is never altered.
"""

import numpy as np

from .efficientdet_postprocessor import EfficientDetPostprocessor


class FastEfficientDetPostprocessor(EfficientDetPostprocessor):
    """Score-gated / format-cached variant of :class:`EfficientDetPostprocessor`."""

    def __init__(self, input_width: int, input_height: int, config: dict = None):
        super().__init__(input_width, input_height, config)
        # Cached BiFPN box-format decision (None until the first frame decides).
        self._coord_is_regression = None

    def _find_score_box_tensors(self, outputs):
        """Locate the box tensor (last dim 4) and score tensors among outputs."""
        boxes_cand = scores_2d = scores_1d = None
        for t in outputs:
            s = np.squeeze(t)
            if s.ndim == 2 and s.shape[-1] == 4 and boxes_cand is None:
                boxes_cand = s
            elif s.ndim == 2 and s.shape[-1] > 4 and scores_2d is None:
                scores_2d = s
            elif s.ndim == 1 and scores_1d is None:
                scores_1d = s
        return boxes_cand, scores_2d, scores_1d

    def _gated_scores_classes(self, scores_2d, scores_1d):
        """Per-anchor max score + class ids, deferring argmax to survivors.

        The max is required to gate; the argmax (the expensive part on ~76k
        anchors x ~90 classes) is computed only for anchors above the score
        threshold and scattered into a zero-filled array. Returns
        ``(None, None)`` when no usable score tensor is present.
        """
        if scores_1d is None and scores_2d is not None:
            if self.has_background and scores_2d.shape[-1] > 1:
                fg = scores_2d[:, 1:]
            else:
                fg = scores_2d
            scores_cand = np.max(fg, axis=1)
            class_ids = np.zeros(len(scores_cand), dtype=int)
            keep = np.nonzero(scores_cand >= self.score_threshold)[0]
            if keep.size:
                class_ids[keep] = np.argmax(fg[keep], axis=1)
            return scores_cand, class_ids
        if scores_1d is not None:
            return scores_1d, np.zeros(len(scores_1d), dtype=int)
        return None, None

    def _is_regression_format(self, boxes_cand):
        """Decide regression-vs-absolute once, then reuse the cached result.

        Identical computation to the standard path on the first frame; a given
        model never changes its output format, so later frames skip the
        ~76k-element ``percentile`` / ``median`` scan.
        """
        if self._coord_is_regression is None:
            box_95pct = np.percentile(np.abs(boxes_cand), 95)
            median_val = np.median(boxes_cand)
            image_size = max(self.input_width, self.input_height)
            self._coord_is_regression = bool(
                box_95pct < image_size * 0.1 and abs(median_val) < 2.0)
        return self._coord_is_regression

    def _process_multi_output(self, outputs, ctx):
        boxes_cand, scores_2d, scores_1d = self._find_score_box_tensors(outputs)
        if boxes_cand is None:
            return []

        scores_cand, class_ids = self._gated_scores_classes(scores_2d, scores_1d)
        if scores_cand is None:
            return []

        n = min(len(boxes_cand), len(scores_cand))
        boxes_cand = boxes_cand[:n]
        scores_cand = scores_cand[:n]
        class_ids = class_ids[:n]

        if self._is_regression_format(boxes_cand):
            return self._decode_anchor_results(boxes_cand, scores_cand, class_ids, ctx)
        return self._decode_results(boxes_cand, scores_cand, class_ids, ctx)

    def get_model_name(self) -> str:
        return "efficientdet_fast"
