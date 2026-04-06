#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Zero-DCE Asynchronous Inference Example

Usage:
    python zero_dce_async_cpp_postprocess.py --model model.dxnn --video input.mp4
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Always use Python fallback — the compiled dx_postprocess.ZeroDCEPostProcess
# does not support 3-channel direct enhanced image output, causing hangs.
from common.processors.cpp_compat import ZeroDCEPostProcess
from common.utility import convert_cpp_zero_dce
from factory import Zero_dceFactory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("Zero-DCE Async Inference")
def main():
    args = parse_args()
    factory = Zero_dceFactory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = ZeroDCEPostProcess(runner.input_width, runner.input_height)
        runner._cpp_convert_fn = convert_cpp_zero_dce

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
