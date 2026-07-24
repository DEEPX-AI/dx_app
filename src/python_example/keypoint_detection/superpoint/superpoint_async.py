#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""SuperPoint Asynchronous Inference Example

Usage:
    python superpoint_async.py --model model.dxnn --video input.mp4
"""
import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import SuperpointFactory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("SuperPoint Async Inference")


def main():
    args = parse_args()
    runner = AsyncRunner(SuperpointFactory())
    runner.run(args)


if __name__ == "__main__":
    main()
