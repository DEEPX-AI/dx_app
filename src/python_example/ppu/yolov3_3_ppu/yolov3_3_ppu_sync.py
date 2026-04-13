#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov3_3_ppu Synchronous Inference Example

Usage:
    python yolov3_3_ppu_sync.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Yolov3_3_ppuFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv5s-PPU Sync Inference")
def main():
    args = parse_args()
    factory = Yolov3_3_ppuFactory()
    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
