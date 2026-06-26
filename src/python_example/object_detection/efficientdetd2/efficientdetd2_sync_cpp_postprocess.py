#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
EfficientDet Synchronous Inference Example

NOTE: EfficientDet requires BiFPN multi-output + anchor regression decoding.
      Falls back to Python postprocessing instead of C++ PostProcess binding.

Usage:
    python efficientdetd2_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
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

from dx_postprocess import EfficientDetPostProcess
from common.utility import convert_cpp_detections
from factory import Efficientdetd2Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("EfficientDet-D2 Sync Inference (C++ Postprocess)")
def main():
    args = parse_args()
    factory = Efficientdetd2Factory()

    def on_engine_init(runner):
        config = runner.factory.config
        score_thr = config.get("score_threshold", config.get("conf_threshold", 0.3))
        nms_thr = config.get("nms_threshold", 0.45)
        num_classes = config.get("num_classes", 90)
        has_bg = config.get("has_background", True)
        runner._cpp_postprocessor = EfficientDetPostProcess(
            runner.input_width, runner.input_height, score_thr, nms_thr, num_classes, has_bg)
        runner._cpp_convert_fn = convert_cpp_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
