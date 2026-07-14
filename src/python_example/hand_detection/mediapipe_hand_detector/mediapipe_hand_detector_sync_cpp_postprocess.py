#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""MediaPipe Hand Detector Synchronous Inference Example (C++ Postprocess)

Usage:
    python mediapipe_hand_detector_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
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
    _dxrt_dir = os.environ.get('DEEPX_SDK_DIR')
    if _dxrt_dir:
        os.add_dll_directory(os.path.join(_dxrt_dir, 'bin'))

from dx_postprocess import MediaPipeHandPostProcess
from common.utility import convert_cpp_mediapipe_hand
from factory import Mediapipe_hand_detectorFactory
from common.runner import SyncRunner, parse_common_args


def parse_args():
    return parse_common_args("MediaPipe Hand Detector Sync Inference")


def main():
    args = parse_args()
    factory = Mediapipe_hand_detectorFactory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = MediaPipeHandPostProcess(runner.input_width)
        runner._cpp_convert_fn = convert_cpp_mediapipe_hand

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
