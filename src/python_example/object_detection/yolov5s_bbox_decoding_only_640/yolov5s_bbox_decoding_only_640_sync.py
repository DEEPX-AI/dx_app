#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov5s_bbox_decoding_only_640 Synchronous Inference Example

Usage:
    python yolov5s_bbox_decoding_only_640_sync.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Yolov5s_bbox_decoding_only_640Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv5n Sync Inference")
def main():
    args = parse_args()
    factory = Yolov5s_bbox_decoding_only_640Factory()
    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
