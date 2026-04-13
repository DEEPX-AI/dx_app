#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov26OBB Asynchronous Inference Example

Usage:
    python yolo26x_obb_async_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import OBBPostProcess
from common.utility import convert_cpp_obb_detections
from factory import Yolo26x_obbFactory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv26n-OBB Async Inference (C++ Postprocess)")
def main():
    args = parse_args()
    factory = Yolo26x_obbFactory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        config = runner.factory.config
        score_thr = config.get("score_threshold", 0.3)
        runner._cpp_postprocessor = OBBPostProcess(input_w, input_h, score_thr)
        runner._cpp_convert_fn = convert_cpp_obb_detections

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
