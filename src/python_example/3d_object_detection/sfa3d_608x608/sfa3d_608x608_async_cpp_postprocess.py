#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
SFA3D Asynchronous Inference with C++ Postprocessing

Usage:
    python sfa3d_608x608_async_cpp_postprocess.py --model model.dxnn --image pointcloud.bin
"""

import sys
import os
import logging
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

if os.name == 'nt':
    _dxrt_dir = os.environ.get('DXRT_DIR')
    if _dxrt_dir:
        os.add_dll_directory(os.path.join(_dxrt_dir, 'bin'))

from factory import Sfa3d608x608Factory
from common.runner import AsyncRunner, parse_common_args
from common.utility.kitti_calib import apply_kitti_companion_dirs_from_args
from calib_policy import enforce_calib_policy

logger = logging.getLogger(__name__)


def parse_args():
    return parse_common_args(
        "SFA3D 608x608 Async Inference (C++ postprocess)",
        include_kitti_paths=True,
    )


def main():
    args = parse_args()
    apply_kitti_companion_dirs_from_args(args)
    enforce_calib_policy(args)
    factory = Sfa3d608x608Factory()

    def on_engine_init(runner):
        try:
            from dx_postprocess import SFA3DPostProcess
            input_w = runner.input_width
            input_h = runner.input_height
            config = runner.factory.config
            score_thr = float(config.get("score_threshold", 0.3))
            nms_thr = float(config.get("nms_threshold", 0.2))
            runner._cpp_postprocessor = SFA3DPostProcess(
                input_w, input_h, score_thr, nms_thr)
            from common.utility import convert_cpp_sfa3d_detections
            runner._cpp_convert_fn = convert_cpp_sfa3d_detections
        except ImportError:
            logger.warning(
                "dx_postprocess.SFA3DPostProcess not available — "
                "falling back to Python postprocessor.")

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
