#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov26OBB Synchronous Inference Example - DX-APP v3.0.0 (C++ Postprocess)

Usage:
    python yolo26n_obb_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
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
from factory import Yolo26n_obbFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv26n-OBB Sync Inference (C++ Postprocess)")
def main():
    args = parse_args()
    factory = Yolo26n_obbFactory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        runner._cpp_postprocessor = OBBPostProcess(input_w, input_h, 0.3)
        runner._cpp_convert_fn = convert_cpp_obb_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
