#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov5 Asynchronous Inference Example

Usage:
    python yolov6n_0_2_1_async_cpp_postprocess.py --model model.dxnn --video input.mp4
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Yolov6n_0_2_1Factory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv6n_0_2_1 Async Inference")
def main():
    args = parse_args()
    factory = Yolov6n_0_2_1Factory()
    runner = AsyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
