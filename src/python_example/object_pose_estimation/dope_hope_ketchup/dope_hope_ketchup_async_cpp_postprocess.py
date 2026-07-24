#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""DOPE Hope-Ketchup Asynchronous Inference Example (C++ Postprocess)

Usage:
    python dope_hope_ketchup_async_cpp_postprocess.py --model model.dxnn --image input.jpg
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

from dx_postprocess import DOPEPostProcess
from common.utility import convert_cpp_dope
from factory import Dope_hope_ketchupFactory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("DOPE Hope-Ketchup Async Inference")


def main():
    args = parse_args()
    factory = Dope_hope_ketchupFactory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = DOPEPostProcess(runner.input_width, runner.input_height)
        runner._cpp_convert_fn = convert_cpp_dope

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
