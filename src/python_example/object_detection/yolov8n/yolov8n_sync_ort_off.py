#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov8 Synchronous Inference Example - DX-APP v3.0.0 (ORT OFF)

Usage:
    python yolov8n_sync_ort_off.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Yolov8Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv8n Sync Inference")
def main():
    args = parse_args()
    factory = Yolov8Factory()
    runner = SyncRunner(factory, use_ort=False)
    runner.run(args)

if __name__ == "__main__":
    main()
