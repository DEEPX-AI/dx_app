#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov10 Asynchronous Inference Example

Usage:
    python yolov10x_async_cpp_postprocess.py --model model.dxnn --video input.mp4
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import YOLOv10PostProcess
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections
from factory import Yolov10xFactory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv10x Async Inference")
def main():
    args = parse_args()
    factory = Yolov10xFactory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        use_ort = InferenceOption().get_use_ort()
        config = runner.factory.config
        score_thr = config.get("score_threshold", 0.3)
        nms_thr = config.get("nms_threshold", 0.45)
        runner._cpp_postprocessor = YOLOv10PostProcess(input_w, input_h, score_thr, nms_thr, use_ort)
        runner._cpp_convert_fn = convert_cpp_detections

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
