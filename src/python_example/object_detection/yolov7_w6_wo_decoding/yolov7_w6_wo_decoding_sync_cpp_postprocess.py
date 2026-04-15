#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov7_w6_wo_decoding Synchronous Inference Example

Usage:
    python yolov7_w6_wo_decoding_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


import os
if os.name == 'nt':
    _dxrt_dir = os.environ.get('DXRT_DIR')
    if _dxrt_dir:
        os.add_dll_directory(os.path.join(_dxrt_dir, 'bin'))

from factory import Yolov7_w6_wo_decodingFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv7 Sync Inference")
def main():
    args = parse_args()
    factory = Yolov7_w6_wo_decodingFactory()

    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
