#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov8 PPU Synchronous Inference Example - DX-APP v3.0.0 (C++ Postprocess)

Usage:
    python yolov8_ppu_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import YOLOv8PPUPostProcess
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections
from factory import Yolov8PpuFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv8-PPU Sync Inference")
def main():
    args = parse_args()
    factory = Yolov8PpuFactory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        runner._cpp_postprocessor = YOLOv8PPUPostProcess(input_w, input_h, 0.4, 0.5)
        runner._cpp_convert_fn = convert_cpp_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
