"""
BT.601 limited-range (studio swing) YCbCr conversions.

OpenCV's ``COLOR_BGR2GRAY`` / ``COLOR_BGR2YCrCb`` use the **full-range**
convention: Y spans [0, 255] with no offset. MATLAB's ``rgb2ycbcr`` — the
convention almost every published super-resolution model is trained with —
uses **limited range**: Y spans [16, 235], Cb/Cr are centred on 128 with a
±112 swing.

Feeding a full-range Y to a model trained on limited-range Y puts the input
outside its training domain: the error is +20 at white and -16 at black. The
two chains are each internally self-consistent, so no global colour cast
appears; the damage shows up only through the network's non-linearities, as
lost detail and sharpness.

These helpers implement the limited-range convention so models trained that
way (ESPCN, and other MATLAB-pipeline SR models) get the input they expect.

Channel order note
------------------
dx_app follows OpenCV and stores the chroma planes as **YCrCb** (Cr before
Cb). These helpers keep that order so they drop into the existing
``np.stack([y, cr, cb])`` call sites unchanged.

References
----------
ITU-R BT.601; MATLAB ``rgb2ycbcr`` / ``ycbcr2rgb``.
"""

import numpy as np

__all__ = [
    "bgr_to_y_limited",
    "bgr_to_ycrcb_limited",
    "ycrcb_limited_to_bgr",
    "Y_LIMITED_MIN",
    "Y_LIMITED_MAX",
]

Y_LIMITED_MIN = 16
Y_LIMITED_MAX = 235

# Forward coefficients on the 0-255 scale (MATLAB rgb2ycbcr / 255).
_KR, _KG, _KB = 65.481 / 255.0, 128.553 / 255.0, 24.966 / 255.0
_CB_R, _CB_G, _CB_B = -37.797 / 255.0, -74.203 / 255.0, 112.0 / 255.0
_CR_R, _CR_G, _CR_B = 112.0 / 255.0, -93.786 / 255.0, -18.214 / 255.0

# Inverse coefficients on the 0-255 scale.
_IY = 255.0 / 219.0            # 1.164383
_IR_CR = 1.596027
_IG_CB, _IG_CR = -0.391762, -0.812968
_IB_CB = 2.017232


def bgr_to_y_limited(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> limited-range Y plane as uint8 in [16, 235].

    This is the drop-in replacement for ``cv2.cvtColor(bgr, COLOR_BGR2GRAY)``
    when the consumer is a model trained on MATLAB ``rgb2ycbcr`` Y.

    Args:
        bgr: [H, W, 3] uint8 BGR image (a [H, W] input is passed through, as
            it is already single-channel).

    Returns:
        [H, W] uint8 Y plane.
    """
    if bgr.ndim == 2:
        return bgr
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    y = Y_LIMITED_MIN + _KR * r + _KG * g + _KB * b
    return np.clip(y, 0.0, 255.0).round().astype(np.uint8)


def bgr_to_ycrcb_limited(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> limited-range [H, W, 3] uint8 in OpenCV **YCrCb** order.

    Drop-in replacement for ``cv2.cvtColor(bgr, COLOR_BGR2YCrCb)``.
    """
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    y = Y_LIMITED_MIN + _KR * r + _KG * g + _KB * b
    cr = 128.0 + _CR_R * r + _CR_G * g + _CR_B * b
    cb = 128.0 + _CB_R * r + _CB_G * g + _CB_B * b

    out = np.stack([y, cr, cb], axis=2)
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


def ycrcb_limited_to_bgr(ycrcb: np.ndarray) -> np.ndarray:
    """Limited-range [H, W, 3] YCrCb uint8 -> BGR uint8.

    Drop-in replacement for ``cv2.cvtColor(ycrcb, COLOR_YCrCb2BGR)``.
    """
    y = ycrcb[:, :, 0].astype(np.float32) - Y_LIMITED_MIN
    cr = ycrcb[:, :, 1].astype(np.float32) - 128.0
    cb = ycrcb[:, :, 2].astype(np.float32) - 128.0

    yy = _IY * y
    r = yy + _IR_CR * cr
    g = yy + _IG_CB * cb + _IG_CR * cr
    b = yy + _IB_CB * cb

    out = np.stack([b, g, r], axis=2)
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)
