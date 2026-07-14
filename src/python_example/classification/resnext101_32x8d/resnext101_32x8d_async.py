#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Resnext101 32x8d Asynchronous Inference Example

Usage:
    python resnext101_32x8d_async.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Resnext101_32x8dFactory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("Resnext101 32x8d Sync Inference")


def main():
    args = parse_args()
    factory = Resnext101_32x8dFactory()
    runner = AsyncRunner(factory)
    runner.run(args)


if __name__ == "__main__":
    main()
