#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov5Face Synchronous Inference Example

Usage:
    python yolov7_face_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common.processors.cpp_compat import PythonFallbackPostProcess
from factory import Yolov7_faceFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv7-Face Sync Inference")
def main():
    args = parse_args()
    factory = Yolov7_faceFactory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = PythonFallbackPostProcess(runner)
        runner._cpp_convert_fn = None

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
