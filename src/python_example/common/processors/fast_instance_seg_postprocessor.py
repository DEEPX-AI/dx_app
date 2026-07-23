"""
Generic fast instance-segmentation postprocessor.

This postprocessor reuses the full detection/NMS pipeline of
``InstanceSegPostprocessor`` and only changes how prototype masks are turned
into per-instance masks.

The standard pipeline, for every kept detection, does:

  1. ``sigmoid(coef @ proto)`` at the low prototype resolution (e.g. 160x160),
  2. bilinear-upsample the *full* mask to the model input resolution
     (e.g. 640x640) -- even though the bbox covers a small fraction of it,
  3. zero everything outside the bbox,
  4. crop letterbox padding and bilinear-resize again to the original image.

Steps 2 and 4 each allocate and resize a full input-resolution float mask per
instance, and step 2 upsamples a large region that step 3 immediately throws
away.

The fast variant follows the well-known "native" mask path:

  1. ``sigmoid(coef @ proto)`` at prototype resolution (same as standard),
  2. crop each mask to its bbox *in prototype coordinates* (cheap, low-res),
  3. remove letterbox padding in prototype coordinates and bilinear-resize the
     prototype-resolution crop **directly** to the original image size (one
     resize, from a small source).

The expensive full input-resolution intermediate is skipped entirely.

Trade-off classification: this is a *mostly lossless* optimization. Inside the
bbox the mask values are essentially identical to the standard path; the only
difference is sub-pixel boundary interpolation introduced by resizing from the
prototype grid instead of the input grid (an A+B "mixed" optimization). The
final binary masks agree at high IoU on typical objects. Adoption stays gated
per-model via the path-B metric check, consistent with the fast_segmentation
policy (never a blanket default conversion).
"""

import cv2
import numpy as np

from .instance_seg_postprocessor import InstanceSegPostprocessor


class FastInstanceSegPostprocessor(InstanceSegPostprocessor):
    """Fast instance-seg postprocessor using prototype-resolution native masks."""

    def _generate_scaled_masks(self, kept_mask_coefs, proto_raw, keep, boxes_x1y1x2y2):
        """Return bbox-cropped masks at *prototype* resolution (no full upsample)."""
        proto = np.squeeze(proto_raw)
        if (proto.ndim == 3 and proto.shape[-1] == self.num_masks
                and proto.shape[0] != self.num_masks):
            proto = np.transpose(proto, (2, 0, 1))  # HWC -> CHW
        c, mh, mw = proto.shape

        masks = 1.0 / (1.0 + np.exp(-(kept_mask_coefs @ proto.reshape(c, -1))))
        masks = masks.reshape(-1, mh, mw).astype(np.float32)

        # Crop each mask to its bbox, expressed in prototype coordinates.
        ratio_w = mw / max(self.input_width, 1)
        ratio_h = mh / max(self.input_height, 1)
        for i, box in enumerate(boxes_x1y1x2y2[keep][:, :4]):
            x1 = int(np.floor(box[0] * ratio_w))
            y1 = int(np.floor(box[1] * ratio_h))
            x2 = int(np.ceil(box[2] * ratio_w))
            y2 = int(np.ceil(box[3] * ratio_h))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mw, x2), min(mh, y2)
            masks[i, :y1, :] = 0
            masks[i, y2:, :] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks

    def _crop_mask_to_original(self, mask_proto, ctx, box=None):
        """Map a prototype-resolution mask to the original image size.

        When the original-space ``box`` is provided (the normal path from
        ``process``), resize only the bbox ROI — crop the prototype mask to the
        box, resize that small region **directly** to the box's original size at
        its aligned origin, and paste into a zero canvas. This avoids the full
        ``(original_h x original_w)`` resize per instance. Aligning the resized
        crop to the box's true original span (rather than squishing an expanded
        proto crop into the exact box) keeps the mask boundary faithful to the
        standard path (measured mask IoU ~0.99 vs the full-resolution path).

        Falls back to the whole-content-region resize when ``box`` is None.
        """
        mh, mw = mask_proto.shape
        ratio_w = mw / max(self.input_width, 1)   # proto px per input px
        ratio_h = mh / max(self.input_height, 1)
        gain = max(ctx.scale, 1e-6)
        pad_x, pad_y = ctx.pad_x, ctx.pad_y
        ow, oh = ctx.original_width, ctx.original_height

        if box is None:
            # Legacy path: resize the padded content region to the full image.
            unpad_w = int(round(ow * gain))
            unpad_h = int(round(oh * gain))
            left = max(0, int(np.floor(pad_x * ratio_w)))
            top = max(0, int(np.floor(pad_y * ratio_h)))
            right = min(mw, int(np.ceil((pad_x + unpad_w) * ratio_w)))
            bottom = min(mh, int(np.ceil((pad_y + unpad_h) * ratio_h)))
            crop = mask_proto[top:bottom, left:right]
            if crop.size == 0:
                return np.zeros((oh, ow), dtype=np.float32)
            return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)

        # --- box-ROI path -------------------------------------------------
        out = np.zeros((oh, ow), dtype=np.float32)
        ox1, oy1, ox2, oy2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        ox1 = max(0, min(ox1, ow)); ox2 = max(0, min(ox2, ow))
        oy1 = max(0, min(oy1, oh)); oy2 = max(0, min(oy2, oh))
        if ox2 <= ox1 or oy2 <= oy1:
            return out

        # original box -> input space -> prototype crop indices (expanded)
        px1 = max(0, int(np.floor((ox1 * gain + pad_x) * ratio_w)))
        py1 = max(0, int(np.floor((oy1 * gain + pad_y) * ratio_h)))
        px2 = min(mw, int(np.ceil((ox2 * gain + pad_x) * ratio_w)))
        py2 = min(mh, int(np.ceil((oy2 * gain + pad_y) * ratio_h)))
        if px2 <= px1 or py2 <= py1:
            return out
        crop = mask_proto[py1:py2, px1:px2]

        # Original-space span the crop covers, and its size (aligned resize).
        cx0 = (px1 / ratio_w - pad_x) / gain
        cy0 = (py1 / ratio_h - pad_y) / gain
        dst_w = max(1, int(round((px2 - px1) / ratio_w / gain)))
        dst_h = max(1, int(round((py2 - py1) / ratio_h / gain)))
        resized = cv2.resize(crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

        # The resized crop lands at original origin (dx0, dy0) spanning
        # dst_w x dst_h. Write directly into `out` only where that paste
        # rectangle intersects the bbox — no full-frame scratch canvas.
        dx0, dy0 = int(round(cx0)), int(round(cy0))
        ix_a = max(dx0, ox1); ix_b = min(dx0 + dst_w, ox2)
        iy_a = max(dy0, oy1); iy_b = min(dy0 + dst_h, oy2)
        if ix_b <= ix_a or iy_b <= iy_a:
            return out
        out[iy_a:iy_b, ix_a:ix_b] = resized[iy_a - dy0:iy_b - dy0,
                                             ix_a - dx0:ix_b - dx0]
        return out

    def get_model_name(self) -> str:
        return "fast_instance_seg"
