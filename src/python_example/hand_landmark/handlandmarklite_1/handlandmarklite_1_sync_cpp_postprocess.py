#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Hand Landmark Lite Synchronous Inference Example (C++ Postprocess)

Usage:
    python handlandmarklite_1_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Handlandmarklite_1Factory
from common.runner import SyncRunner, parse_common_args

import os
if os.name == 'nt':
    _dxrt_dir = os.environ.get('DXRT_DIR')
    if _dxrt_dir:
        os.add_dll_directory(os.path.join(_dxrt_dir, 'bin'))

from dx_postprocess import HandLandmarkPostProcess
from common.utility import convert_cpp_hand_landmark

def parse_args():
    return parse_common_args("HandLandmarkLite Sync Inference (C++ PP)")
def main():
    args = parse_args()
    factory = Handlandmarklite_1Factory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = HandLandmarkPostProcess(runner.input_width, runner.input_height)
        runner._cpp_convert_fn = convert_cpp_hand_landmark

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
