#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Efficientnet Synchronous Inference Example - DX-APP v3.0.0

Usage:
    python efficientnet_lite1_sync.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Efficientnet_lite1Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("EfficientnetLite1 Sync Inference")
def main():
    args = parse_args()
    factory = Efficientnet_lite1Factory()
    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
