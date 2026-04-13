#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
SSD Asynchronous Inference Example

Usage:
    python ssdmv1_async_cpp_postprocess.py --model model.dxnn --video input.mp4
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import SSDPostProcess
from common.utility import convert_cpp_detections
from factory import Ssdmv1Factory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("SSD Asyn Inference (C++ Postprocess)")
def main():
    args = parse_args()
    factory = Ssdmv1Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        config = runner.factory.config
        score_thr = config.get("score_threshold", 0.3)
        nms_thr = config.get("nms_threshold", 0.45)
        runner._cpp_postprocessor = SSDPostProcess(
            input_w, input_h, score_thr, nms_thr, 21, True
        )
        runner._cpp_convert_fn = convert_cpp_detections

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
