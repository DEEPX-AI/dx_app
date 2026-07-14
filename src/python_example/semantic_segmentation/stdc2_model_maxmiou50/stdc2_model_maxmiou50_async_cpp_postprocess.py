#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Stdc2 model maxmiou50 Asynchronous Inference Example

Usage:
    python stdc2_model_maxmiou50_async_cpp_postprocess.py --model model.dxnn --image input.jpg
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

from dx_postprocess import SemanticSegPostProcess
from common.utility import convert_cpp_semantic_seg
from factory import Stdc2_model_maxmiou50Factory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("Stdc2 model maxmiou50 Async Inference")


def main():
    args = parse_args()
    factory = Stdc2_model_maxmiou50Factory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = SemanticSegPostProcess(runner.input_width, runner.input_height)
        runner._cpp_convert_fn = convert_cpp_semantic_seg

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
