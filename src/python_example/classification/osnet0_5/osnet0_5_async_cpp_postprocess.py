#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Classification Asynchronous Inference Example

Usage:
    python osnet0_5_async_cpp_postprocess.py --model model.dxnn --image input.jpg
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
    from dx_postprocess import ClassificationPostProcess
except (ImportError, AttributeError):
    from common.processors.cpp_compat import ClassificationPostProcess
from common.utility import convert_cpp_classification
from factory import Osnet0_5Factory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("OSNet-0.5 Async Inference")
def main():
    args = parse_args()
    factory = Osnet0_5Factory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = ClassificationPostProcess()
        runner._cpp_convert_fn = convert_cpp_classification

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
