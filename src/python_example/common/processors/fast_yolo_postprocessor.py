"""Fast YOLOv5/YOLOv7 raw-decode postprocessor (exact, opt-in).

The standard :class:`YOLOv5Postprocessor` decodes raw NPU feature maps by running
``sigmoid`` over the *entire* per-anchor field grid -- including the full class
block ``[A, num_classes, H, W]`` -- for every anchor, and only afterwards drops
~99% of those anchors by the objectness threshold inside ``process()``. Profiling
on real NPU output (e.g. ``yolov7_w6-wo_decoding`` at 1280x1280 with P3-P6 heads)
showed that this full-grid ``sigmoid`` (plus its ``np.clip``) dominates the
postprocess cost.

This variant gates anchors by objectness *before* computing the class sigmoid.
Objectness gating is exact: ``sigmoid`` is monotonic, so the survivor set selected
by ``sigmoid(obj_logit) >= obj_threshold`` is identical to the standard path, and
the box/class decode for the survivors uses the *same* formulas on the *same*
inputs. The produced detections are therefore byte-identical to the standard path
(verified by the parity unit tests) -- unlike the path-B mask approximations, this
is a lossless optimization. It is still gated behind ``--fast-postprocess`` so the
default path is never altered.

Already-decoded single-tensor / 2-tensor outputs (``[1, N, 5+C]`` etc.) are handled
by the unchanged base ``process()`` -- there is no raw grid to gate, so the fast
class behaves exactly like the standard one for those models.
"""

import numpy as np

from .yolo_postprocessor import YOLOv5Postprocessor


class FastYOLOv5Postprocessor(YOLOv5Postprocessor):
    """Objectness-gated raw decode variant of :class:`YOLOv5Postprocessor`."""

    def _gather_scale(self, sel_logits, a_idx, y_idx, x_idx, grid_x, grid_y,
                      stride, anchors):
        """Decode the survivor anchors of a single scale into ``[M, 5+C]`` rows.

        ``sel_logits`` is ``[M, num_fields]`` raw logits gathered for the ``M``
        anchors that already passed the objectness gate; the returned rows match
        the standard decode formulas exactly (same sigmoid, same anchor scaling),
        only restricted to the survivors.
        """
        tx = self._sigmoid(sel_logits[:, 0])
        ty = self._sigmoid(sel_logits[:, 1])
        tw = self._sigmoid(sel_logits[:, 2])
        th = self._sigmoid(sel_logits[:, 3])

        cx = (tx * 2.0 - 0.5 + grid_x) * stride
        cy = (ty * 2.0 - 0.5 + grid_y) * stride

        anchor_w = np.array([a[0] for a in anchors], dtype=np.float32)[a_idx]
        anchor_h = np.array([a[1] for a in anchors], dtype=np.float32)[a_idx]
        w = (tw * 2.0) ** 2 * anchor_w
        h = (th * 2.0) ** 2 * anchor_h

        obj = self._sigmoid(sel_logits[:, 4])
        cls = self._sigmoid(sel_logits[:, 5:5 + self.num_classes])

        out = np.empty((sel_logits.shape[0], 5 + self.num_classes), dtype=np.float32)
        out[:, 0] = cx
        out[:, 1] = cy
        out[:, 2] = w
        out[:, 3] = h
        out[:, 4] = obj
        out[:, 5:] = cls
        return out

    def _empty(self):
        return np.zeros((0, 5 + self.num_classes), dtype=np.float32)

    def _decode_multi_scale_outputs(self, outputs: list) -> np.ndarray:
        """Objectness-gated version of the 4D NCHW/NHWC multi-scale decode."""
        num_fields = 5 + self.num_classes
        num_anchors = 3
        # Same ordering as the standard path so cross-scale concatenation order
        # (and therefore NMS tie-breaking) is preserved.
        sorted_outputs = sorted(outputs, key=lambda t: t.size, reverse=True)

        all_detections = []
        for tensor in sorted_outputs:
            data = np.squeeze(tensor)
            if data.shape[0] == num_anchors * num_fields:
                pass  # NCHW: [C, H, W]
            elif data.shape[-1] == num_anchors * num_fields:
                data = data.transpose(2, 0, 1)  # NHWC -> NCHW

            grid_h, grid_w = data.shape[1], data.shape[2]
            stride = self.input_height // grid_h
            if stride not in self.ANCHORS:
                continue
            anchors = self.ANCHORS[stride]

            data = data.reshape(num_anchors, num_fields, grid_h, grid_w)

            # Objectness gate BEFORE the expensive class sigmoid.
            obj = self._sigmoid(data[:, 4, :, :])          # [A, H, W]
            keep = obj >= self.obj_threshold
            if not np.any(keep):
                continue
            a_idx, y_idx, x_idx = np.nonzero(keep)          # C-order == standard
            sel = data[a_idx, :, y_idx, x_idx]              # [M, num_fields]

            all_detections.append(self._gather_scale(
                sel, a_idx, y_idx, x_idx,
                x_idx.astype(np.float32), y_idx.astype(np.float32),
                stride, anchors))

        if not all_detections:
            return self._empty()
        return np.concatenate(all_detections, axis=0)

    def _decode_raw_multi_scale(self, outputs: list) -> np.ndarray:
        """Objectness-gated version of the 5D ``[1, A, H, W, C]`` raw decode."""
        sorted_outputs = sorted(outputs, key=lambda t: t.shape[2] * t.shape[3],
                                reverse=True)

        all_detections = []
        for tensor in sorted_outputs:
            data = tensor[0]                                # [A, H, W, C]
            grid_h = data.shape[1]
            stride = self.input_height // grid_h
            if stride not in self.ANCHORS:
                continue
            anchors = self.ANCHORS[stride]

            data = data.transpose(0, 3, 1, 2)               # [A, C, H, W]

            obj = self._sigmoid(data[:, 4, :, :])           # [A, H, W]
            keep = obj >= self.obj_threshold
            if not np.any(keep):
                continue
            a_idx, y_idx, x_idx = np.nonzero(keep)
            sel = data[a_idx, :, y_idx, x_idx]              # [M, num_fields]

            all_detections.append(self._gather_scale(
                sel, a_idx, y_idx, x_idx,
                x_idx.astype(np.float32), y_idx.astype(np.float32),
                stride, anchors))

        if not all_detections:
            return self._empty()
        return np.concatenate(all_detections, axis=0)

    def get_model_name(self) -> str:
        return "yolov5_fast"
