"""Fast YOLACT instance-segmentation postprocessor (opt-in approximation).

Keeps the full YOLACT detection / Fast-NMS pipeline of
:class:`YOLACTPostprocessor` unchanged and only changes the mask path. The
standard path upsamples every prototype mask to the model input resolution and
then resizes again to the original image (two large resizes per instance); this
variant crops each mask to its bbox at *prototype* resolution and resizes that
small crop once to the original image.

This is the same path-B approximation that
:class:`FastInstanceSegPostprocessor` applies to YOLOv8-seg. The final binary
masks differ from the standard path only at sub-pixel boundaries, so this is
never the default and is enabled only via ``--fast-postprocess``.
"""

import cv2
import numpy as np

from .yolact_postprocessor import YOLACTPostprocessor


class FastYOLACTPostprocessor(YOLACTPostprocessor):
    """ROI-crop variant of :class:`YOLACTPostprocessor` (prototype-res masks)."""

    def _scale_masks_and_crop(self, masks, boxes_pixel_keep):
        """Crop each mask to its bbox at *prototype* resolution (no input upsample).

        ``masks`` are ``[K, mh, mw]`` prototype-resolution sigmoids. The standard
        path resizes every mask to ``[input_h, input_w]`` first; here we keep the
        small prototype grid and only zero out everything outside the bbox mapped
        into prototype coordinates. The single resize to original resolution then
        happens in :meth:`_to_original_coords`.
        """
        masks = np.ascontiguousarray(masks, dtype=np.float32)
        k = len(masks)
        if k == 0:
            return masks
        mh, mw = masks.shape[1], masks.shape[2]
        ratio_w = mw / self.input_width
        ratio_h = mh / self.input_height
        out = masks.copy()
        for i, box in enumerate(boxes_pixel_keep):
            bx1 = max(0, int(box[0] * ratio_w))
            by1 = max(0, int(box[1] * ratio_h))
            bx2 = min(mw, int(np.ceil(box[2] * ratio_w)))
            by2 = min(mh, int(np.ceil(box[3] * ratio_h)))
            out[i, :by1, :] = 0
            out[i, by2:, :] = 0
            out[i, :, :bx1] = 0
            out[i, :, bx2:] = 0
        return out

    def _to_original_coords(self, box, scaled_mask_i, ctx):
        """Map box back to original space and resize the prototype mask once.

        ``scaled_mask_i`` is at prototype resolution (see
        :meth:`_scale_masks_and_crop`), so letterbox padding — expressed in input
        pixels on ``ctx`` — is rescaled into prototype coordinates before cropping.
        The box mapping is delegated to the shared :meth:`_box_to_original` so the
        coordinates stay byte-identical to the standard path.
        """
        box = self._box_to_original(box, ctx)
        gain = max(ctx.scale, 1e-6)
        pad_x, pad_y = ctx.pad_x, ctx.pad_y
        ow, oh = ctx.original_width, ctx.original_height
        mh, mw = scaled_mask_i.shape[:2]

        if pad_x == 0 and pad_y == 0:
            orig_mask = cv2.resize(scaled_mask_i, (ow, oh), interpolation=cv2.INTER_LINEAR)
            return box, orig_mask

        ratio_w = mw / self.input_width
        ratio_h = mh / self.input_height
        px = int(round(pad_x * ratio_w))
        py = int(round(pad_y * ratio_h))
        unpad_w = int(round(ow * gain * ratio_w))
        unpad_h = int(round(oh * gain * ratio_h))
        m_crop = scaled_mask_i[py:py + unpad_h, px:px + unpad_w]
        orig_mask = (cv2.resize(m_crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
                     if m_crop.size > 0
                     else np.zeros((oh, ow), dtype=np.float32))
        return box, orig_mask

    def get_model_name(self) -> str:
        return "fast_yolact"
