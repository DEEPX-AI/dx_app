#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Common argument parser for DX-APP inference examples.

Consolidates per-model ``parse_args()`` into one reusable function so that
camera, RTSP, save, loop, and dump-tensors options are available uniformly
across **all** models without touching each thin wrapper individually.

Usage (from any model wrapper)::

    from common.runner import parse_common_args

    # Minimal — description only
    args = parse_common_args("YOLOv5 Sync Inference")

    # SR / depth / denoising models that need --output
    args = parse_common_args("ESPCN Sync Inference", include_output=True)
"""

import argparse
import sys


# Stream-input flags that image-only examples do NOT register. Kept here so the
# custom parser below can recognise them in an "unrecognized arguments" error.
_STREAM_FLAGS = ("--video", "-v", "--camera", "-c", "--rtsp", "-r")


class _ImageOnlyArgumentParser(argparse.ArgumentParser):
    """Parser for image-only examples (embedding, ReID, attribute recognition).

    Stream flags (``--video`` / ``--camera`` / ``--rtsp``) are intentionally NOT
    registered so they stay hidden from ``--help``. When a user
    passes one anyway, argparse would report only a generic
    "unrecognized arguments" error. We override :meth:`error` so that an
    explicit, always-shown note explaining the example is image-only is emitted
    as well — while preserving the standard message text and exit code 2.
    """

    def error(self, message):  # noqa: D401 - argparse hook
        if "unrecognized arguments" in message and any(
            flag in message.split() for flag in _STREAM_FLAGS
        ):
            self.print_usage(sys.stderr)
            self.exit(
                2,
                f"{self.prog}: error: {message}\n"
                f"{self.prog}: note: this example is image-only — video/camera/"
                f"RTSP input (--video/--camera/--rtsp) is not supported. "
                f"Use --image (-i) to provide an image file or directory.\n",
            )
        super().error(message)



def parse_common_args(
    description: str = "DX-APP Inference",
    *,
    include_output: bool = False,
    include_stream_inputs: bool = True,
    include_kitti_paths: bool = False,
) -> argparse.Namespace:
    """Parse common inference arguments.

    Standard options (always available):

    ===============  ========================================
    Option           Description
    ===============  ========================================
    --model, -m      Model path (``.dxnn``)
    --image, -i      Input image **or directory** path
    --video, -v      Input video path
    --camera, -c     Camera device ID (e.g. ``0``)
    --rtsp, -r       RTSP stream URL
    --display        Show output window (default on)
    --no-display     Disable display output
    --save, -s       Save output frames / images
    --save-dir       Output save directory (default: auto)
    --loop, -l       Inference loop count (default 1, bare --loop = 2)
    --dump-tensors   Dump raw inference tensors
    --config         Path to config.json (auto-detected)
    --sr-tile-halo   Tile overlap (px) for tiled super-resolution
    ===============  ========================================

    Args:
        description: ``argparse`` description string.
        include_output: If ``True``, add ``--output`` argument
            (used by super-resolution / depth / denoising models).
        include_stream_inputs: If ``False``, omit ``--video``, ``--camera``,
            and ``--rtsp`` for image-only models such as embedding and ReID.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser_cls = (
        argparse.ArgumentParser if include_stream_inputs
        else _ImageOnlyArgumentParser
    )
    parser = parser_cls(description=description, allow_abbrev=False)

    # ---- Model path ----
    # Optional (SDKREQ-529): when omitted, the runner resolves this example's
    # default model from config/model_registry.json and auto-downloads it.
    # An explicitly-given path that is missing errors out without auto-download.
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Model path (.dxnn). If omitted, use this example's default model."
    )

    # ---- Input source (mutually exclusive) ----
    # Image-only tasks (embedding, ReID, …) pass ``include_stream_inputs=False``
    # so ``--video`` / ``--camera`` / ``--rtsp`` are NOT registered at all: they
    # are absent from ``--help`` and argparse rejects them with
    # "unrecognized arguments" instead of accepting-then-refusing at runtime.
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--image", "-i", type=str, default=None,
        help="Input image path or directory (default: task-appropriate sample)"
    )
    if include_stream_inputs:
        input_group.add_argument(
            "--video", "-v", type=str, default=None, help="Input video path"
        )
        input_group.add_argument(
            "--camera", "-c", type=int, default=None, help="Camera device ID (e.g. 0)"
        )
        input_group.add_argument(
            "--rtsp", "-r", type=str, default=None, help="RTSP stream URL"
        )

    # ---- Display ----
    parser.add_argument(
        "--display", action="store_true", default=True,
        help="Show output window (default: True)",
    )
    parser.add_argument(
        "--no-display", dest="display", action="store_false",
        help="Disable display output",
    )

    # ---- Save / Loop / Dump ----
    parser.add_argument(
        "--save", "-s", action="store_true", default=False,
        help="Save output frames / images",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Output save directory (default: artifacts/python_example/)",
    )
    parser.add_argument(
        "--loop", "-l", type=int, nargs="?", const=2, default=1,
        help="Number of inference loops (default: 1). "
             "Use --loop without a value for 2 loops, or --loop N.",
    )
    parser.add_argument(
        "--dump-tensors", action="store_true", default=False,
        help="Dump raw inference tensors for debugging",
    )

    # ---- Config ----
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.json (auto-detected if omitted)",
    )

    # ---- Verbosity ----
    parser.add_argument(
        "--show-log", action="store_true", default=False,
        help="Show detailed per-frame/image [DXAPP] [INFO] logs (default: quiet)",
    )

    # ---- Opt-in fast postprocess ----
    parser.add_argument(
        "--fast-postprocess", action="store_true", default=False,
        help="Use the opt-in fast postprocessor variant when the model "
             "provides one. Exact / byte-identical for detection models "
             "(YOLOv5/v7, EfficientDet, verified by parity tests); approximate "
             "(differs only at sub-pixel mask boundaries) for segmentation "
             "models (instance-seg, YOLACT, SegFormer). The standard path is "
             "always the default.",
    )

    # ---- Tiled super-resolution ----
    # Only the tiled SR path reads this (single-channel SR such as ESPCN, which is
    # compiled at a small fixed input and therefore covers a larger image by
    # tiling). Other models accept and ignore it, the same way --fast-postprocess
    # is safe everywhere. Registered unconditionally because the SR entry points
    # do not pass ``include_output`` consistently, and gating on it would drop the
    # option from some of them.
    parser.add_argument(
        "--sr-tile-halo", type=int, default=None, metavar="PX",
        help="Tile overlap in LR pixels for tiled super-resolution, 0..4 "
             "(default: 4 = the ESPCN receptive-field radius, the most context an "
             "output pixel can use; 0 = no overlap, fastest but seams appear). "
             "Overrides config.json 'sr_tile_halo' and DXAPP_SR_TILE_HALO.",
    )

    # ---- Optional: output path (SR / depth / denoising) ----
    if include_output:
        parser.add_argument(
            "--output", "-o", type=str, help="Output file path"
        )

    # ---- Optional: KITTI companion directories (SFA3D 3D detection) ----
    if include_kitti_paths:
        parser.add_argument(
            "--calib-dir", type=str, default=None,
            help="Directory of {frame_id}.txt calib files paired with --image stems",
        )
        parser.add_argument(
            "--image2-dir", type=str, default=None,
            help="Directory of {frame_id}.png/.jpg camera images paired with --image stems",
        )

    return parser.parse_args()
