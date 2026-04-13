#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
CenterPose Synchronous Inference Example

Usage:
    python centerpose_regnetx_1_6gf_fpn_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import CenterPosePostProcess
from common.utility import convert_cpp_pose_detections
from factory import Centerpose_regnetx_1_6gf_fpnFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("CenterPose Sync Inference (C++ Postprocess)")
def main():
    args = parse_args()
    factory = Centerpose_regnetx_1_6gf_fpnFactory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        config = runner.factory.config
        score_thr = config.get("score_threshold", 0.3)
        nms_thr = config.get("nms_threshold", 0.45)
        runner._cpp_postprocessor = CenterPosePostProcess(
            input_w, input_h, score_thr, nms_thr
        )
        runner._cpp_convert_fn = convert_cpp_pose_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
