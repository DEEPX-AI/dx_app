#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
DeepLabV3+ DRN 512x512 Synchronous Inference Example

Usage:
    python deeplabv3plus_drn_512x512_sync.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Deeplabv3plus_drn_512x512Factory
from common.runner import SyncRunner, parse_common_args


def parse_args():
    return parse_common_args("DeepLabV3+ DRN 512x512 Sync Inference")


def main():
    args = parse_args()
    factory = Deeplabv3plus_drn_512x512Factory()
    runner = SyncRunner(factory)
    runner.run(args)


if __name__ == "__main__":
    main()
