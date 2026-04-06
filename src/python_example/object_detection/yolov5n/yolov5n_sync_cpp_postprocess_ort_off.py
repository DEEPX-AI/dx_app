#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov5 Synchronous Inference Example

Usage:
    python yolov5n_sync_cpp_postprocess_ort_off.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import YOLOv5PostProcess
from common.utility import convert_cpp_detections
from factory import Yolov5Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv5n Sync Inference")
def main():
    args = parse_args()
    factory = Yolov5Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        runner._cpp_postprocessor = YOLOv5PostProcess(input_w, input_h, 0.25, 0.3, 0.45, False)
        runner._cpp_convert_fn = convert_cpp_detections

    runner = SyncRunner(factory, use_ort=False, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
