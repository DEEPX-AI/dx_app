#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
DnCNN Asynchronous Inference Example

NOTE: C++ PostProcess binding is not yet implemented for this model.
      Falls back to Python postprocessing.

Note: DnCNN requires the normalized input image to compute denoised output
      (denoised = input - residual), which the C++ postprocess API cannot
      provide. This file uses the Python postprocessor as fallback.

Usage:
    python dncnn_15_async_cpp_postprocess.py --model model.dxnn --video input.mp4
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Dncnn_15Factory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("DnCNN-15 Async Inference (Python Postprocess)")
def main():
    args = parse_args()
    factory = Dncnn_15Factory()
    # Python fallback: DnCNN postprocess needs ctx.normalized_input
    # which is not accessible from C++ postprocess API
    runner = AsyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
