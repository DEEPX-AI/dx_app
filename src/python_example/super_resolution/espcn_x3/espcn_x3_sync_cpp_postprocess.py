#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
ESPCN Synchronous Inference Example

Note: DnCNN requires the normalized input image to compute denoised output
      (denoised = input - residual), which the C++ postprocess API cannot
      provide. This file uses the Python postprocessor as fallback.

Usage:
    python espcn_x3_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
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

from dx_postprocess import ESPCNPostProcess
from common.utility import convert_cpp_super_resolution
from factory import Espcn_x3Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("ESPCN Sync Inference (Python Postprocess)", include_output=True)
def main():
    args = parse_args()
    factory = Espcn_x3Factory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = ESPCNPostProcess(
            runner.input_width, runner.input_height)
        runner._cpp_convert_fn = convert_cpp_super_resolution

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
