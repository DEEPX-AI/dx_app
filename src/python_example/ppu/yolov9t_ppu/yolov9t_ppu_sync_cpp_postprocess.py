#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
YOLOv9T PPU Synchronous Inference Example

Usage:
    python yolov9t_ppu_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


import os
if os.name == 'nt':
    _dxrt_dir = os.environ.get('DEEPX_SDK_DIR')
    if _dxrt_dir:
        os.add_dll_directory(os.path.join(_dxrt_dir, 'bin'))

from dx_postprocess import YOLOv8PPUPostProcess
from common.utility import convert_cpp_detections
from factory import Yolov9tPpuFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv9T-PPU Sync Inference")
def main():
    args = parse_args()
    factory = Yolov9tPpuFactory()

    def on_engine_init(runner):
        config = runner.factory.config
        score_thr = config.get("conf_threshold", config.get("score_threshold", 0.4))
        nms_thr = config.get("nms_threshold", 0.5)
        runner._cpp_postprocessor = YOLOv8PPUPostProcess(
            runner.input_width, runner.input_height, score_thr, nms_thr)
        runner._cpp_convert_fn = convert_cpp_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
