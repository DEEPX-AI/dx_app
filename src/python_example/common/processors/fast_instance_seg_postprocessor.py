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

    def _crop_mask_to_original(self, mask_proto, ctx):
        """Map a prototype-resolution mask directly to the original image size."""
        mh, mw = mask_proto.shape
        ratio_w = mw / max(self.input_width, 1)
        ratio_h = mh / max(self.input_height, 1)

        gain = max(ctx.scale, 1e-6)
        unpad_w = int(round(ctx.original_width * gain))
        unpad_h = int(round(ctx.original_height * gain))

        # Letterbox-padded content region, expressed in prototype coordinates.
        left = int(np.floor(ctx.pad_x * ratio_w))
        top = int(np.floor(ctx.pad_y * ratio_h))
        right = int(np.ceil((ctx.pad_x + unpad_w) * ratio_w))
        bottom = int(np.ceil((ctx.pad_y + unpad_h) * ratio_h))
        left, top = max(0, left), max(0, top)
        right, bottom = min(mw, right), min(mh, bottom)

        crop = mask_proto[top:bottom, left:right]
        if crop.size == 0:
            return np.zeros((ctx.original_height, ctx.original_width), dtype=np.float32)
        return cv2.resize(
            crop, (ctx.original_width, ctx.original_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def get_model_name(self) -> str:
        return "fast_instance_seg"
