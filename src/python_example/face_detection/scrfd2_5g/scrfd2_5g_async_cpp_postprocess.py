#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
SCRFD Asynchronous Inference Example

Usage:
    python scrfd2_5g_async_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import SCRFDPostProcess
from dx_engine import InferenceOption
from common.utility import convert_cpp_face_detections
from factory import Scrfd2_5gFactory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("SCRFD-2_5G Async Inference")
def main():
    args = parse_args()
    factory = Scrfd2_5gFactory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        use_ort = InferenceOption().get_use_ort()
        config = runner.factory.config
        score_thr = config.get("score_threshold", 0.3)
        nms_thr = config.get("nms_threshold", 0.45)
        runner._cpp_postprocessor = SCRFDPostProcess(input_w, input_h, score_thr, nms_thr, use_ort)
        runner._cpp_convert_fn = convert_cpp_face_detections

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
