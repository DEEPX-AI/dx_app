#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Face Alignment Synchronous Inference Example (C++ Postprocess)

Usage:
    python 3ddfa_v2_mobilnetv1_120x120_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
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

try:
    from dx_postprocess import Face3DPostProcess
except (ImportError, AttributeError):
    from common.processors.cpp_compat import Face3DPostProcess
from common.utility import convert_cpp_face3d
from factory import N3ddfa_v2_mobilnetv1_120x120Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("3DDFA-v2-MobileNetV1 Sync Inference (C++ PP)")
def main():
    args = parse_args()
    factory = N3ddfa_v2_mobilnetv1_120x120Factory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = Face3DPostProcess(runner.input_width, runner.input_height)
        runner._cpp_convert_fn = convert_cpp_face3d

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
