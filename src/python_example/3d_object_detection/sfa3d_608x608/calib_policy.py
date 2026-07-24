#!/usr/bin/env python3
"""Calibration policy helpers for SFA3D examples."""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Optional

from common.utility.kitti_calib import candidate_calib_paths

logger = logging.getLogger(__name__)


def load_require_calib(args) -> bool:
    """Return require_calib flag from config file (default: False)."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        config_path = str(Path(__file__).with_name("config.json"))
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return bool(cfg.get("require_calib", False))
    except Exception:
        return False


def _find_reference_bin(input_path: str) -> Optional[Path]:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".bin":
        return p
    if p.is_dir():
        bins = sorted(glob.glob(str(p / "*.bin")))
        if bins:
            return Path(bins[0])
    return None


def enforce_calib_policy(args) -> None:
    """Emit required calibration warning/error codes per spec."""
    input_path = getattr(args, "image", None)
    if not input_path:
        return

    ref_bin = _find_reference_bin(input_path)
    if ref_bin is None:
        return

    has_calib = any(c.exists() for c in candidate_calib_paths(ref_bin))
    if has_calib:
        return

    require_calib = load_require_calib(args)
    if require_calib:
        raise FileNotFoundError(
            f"SFA3D-E-CALIB-MISSING: calibration file not found for {ref_bin.name}"
        )

    logger.warning(
        "SFA3D-W-CALIB-MISSING: calibration file not found; running in BEV-only mode."
    )

