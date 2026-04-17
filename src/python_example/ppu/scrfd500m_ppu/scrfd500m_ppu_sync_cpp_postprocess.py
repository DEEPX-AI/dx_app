#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
SCRFD500M PPU Synchronous Inference Example

Usage:
    python scrfd500m_ppu_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
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
    _dxrt_dir = os.environ.get('DXRT_DIR')
    if _dxrt_dir:
        os.add_dll_directory(os.path.join(_dxrt_dir, 'bin'))

from dx_postprocess import SCRFDPPUPostProcess
from dx_engine import InferenceOption
from common.utility import convert_cpp_face_detections
from factory import Scrfd500mPpuFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("SCRFD500M-PPU Sync Inference")
def main():
    args = parse_args()
    factory = Scrfd500mPpuFactory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        config = runner.factory.config
        score_thr = config.get("score_threshold", 0.5)
        nms_thr = config.get("nms_threshold", 0.45)
        runner._cpp_postprocessor = SCRFDPPUPostProcess(input_w, input_h, score_thr, nms_thr)
        runner._cpp_convert_fn = convert_cpp_face_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
