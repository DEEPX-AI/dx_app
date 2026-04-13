#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Face Detection Synchronous Inference Example

Usage:
    python retinaface_mobilenet0_25_640_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import RetinaFacePostProcess
from common.utility import convert_cpp_face_detections
from factory import Retinaface_mobilenet0_25_640Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("RetinaFace-MobileNet Sync Inference (C++ Postprocess)")
def main():
    args = parse_args()
    factory = Retinaface_mobilenet0_25_640Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        config = runner.factory.config
        score_thr = config.get("score_threshold", 0.5)
        nms_thr = config.get("nms_threshold", 0.4)
        runner._cpp_postprocessor = RetinaFacePostProcess(input_w, input_h, score_thr, nms_thr)
        runner._cpp_convert_fn = convert_cpp_face_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
