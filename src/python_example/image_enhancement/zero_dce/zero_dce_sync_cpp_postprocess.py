#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Zero-DCE Synchronous Inference Example

Usage:
    python zero_dce_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Use native C++ postprocessor from dx_postprocess.so

import os
if os.name == 'nt':
    _dxrt_dir = os.environ.get('DXRT_DIR')
    if _dxrt_dir:
        os.add_dll_directory(os.path.join(_dxrt_dir, 'bin'))

from dx_postprocess import ZeroDCEPostProcess
from common.utility import convert_cpp_zero_dce
from factory import Zero_dceFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("Zero-DCE Sync Inference")
def main():
    args = parse_args()
    factory = Zero_dceFactory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = ZeroDCEPostProcess(runner.input_width, runner.input_height)
        runner._cpp_convert_fn = convert_cpp_zero_dce

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
