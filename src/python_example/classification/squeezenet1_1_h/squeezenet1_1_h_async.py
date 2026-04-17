#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Squeezenet1_1_h Asynchronous Inference Example

Usage:
    python squeezenet1_1_h_async.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Squeezenet1_1_hFactory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("EfficientNet-Lite0 Async Inference")
def main():
    args = parse_args()
    factory = Squeezenet1_1_hFactory()
    runner = AsyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
