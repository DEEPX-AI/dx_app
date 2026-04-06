#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
SSDMV1 Synchronous Inference Example

Usage:
    python ssdmv2lite_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from dx_postprocess import SSDPostProcess
    from common.utility import convert_cpp_detections
except ImportError:
    SSDPostProcess = None
from factory import Ssdmv2liteFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("SSDMV2Lite Sync Inference (C++ Postprocess)")
def main():
    args = parse_args()
    factory = Ssdmv2liteFactory()

    def on_engine_init(runner):
        if SSDPostProcess is not None:
            input_w = runner.input_width
            input_h = runner.input_height
            runner._cpp_postprocessor = SSDPostProcess(
                input_w, input_h, 0.3, 0.45, 20, True
            )
            runner._cpp_convert_fn = convert_cpp_detections
        else:
            from common.processors.cpp_compat import PythonFallbackPostProcess
            runner._cpp_postprocessor = PythonFallbackPostProcess(runner)
            runner._cpp_convert_fn = None

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
