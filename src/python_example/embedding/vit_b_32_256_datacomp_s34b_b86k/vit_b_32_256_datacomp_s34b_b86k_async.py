#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Vit b 32 256 datacomp s34b b86k Asynchronous Inference Example

Usage:
    python vit_b_32_256_datacomp_s34b_b86k_async.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Vit_b_32_256_datacomp_s34b_b86kFactory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("Vit b 32 256 datacomp s34b b86k Async Inference", include_stream_inputs=False)


def main():
    args = parse_args()
    factory = Vit_b_32_256_datacomp_s34b_b86kFactory()
    runner = AsyncRunner(factory)
    runner.run(args)


if __name__ == "__main__":
    main()
