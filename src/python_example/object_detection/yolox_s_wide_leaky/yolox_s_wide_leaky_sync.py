#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
YOLOX Synchronous Inference Example - DX-APP v3.0.0

Usage:
    python yolox_s_wide_leaky_sync.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Yolox_s_wide_leakyFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOX_S_WIDE_LEAKY Sync Inference")
def main():
    args = parse_args()
    factory = Yolox_s_wide_leakyFactory()
    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
