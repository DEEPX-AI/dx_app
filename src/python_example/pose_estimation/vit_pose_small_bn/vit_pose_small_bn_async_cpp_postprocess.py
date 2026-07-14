#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""VitPose Small BN Asynchronous Inference Example (C++ Postprocess)

Usage:
    python vit_pose_small_bn_async_cpp_postprocess.py --model model.dxnn --image input.jpg
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

from dx_postprocess import VitPosePostProcess
from common.utility import convert_cpp_vitpose
from factory import Vit_pose_small_bnFactory
from common.runner import AsyncRunner, parse_common_args


def parse_args():
    return parse_common_args("VitPose Small BN Async Inference")


def main():
    args = parse_args()
    factory = Vit_pose_small_bnFactory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = VitPosePostProcess(runner.input_width, runner.input_height)
        runner._cpp_convert_fn = convert_cpp_vitpose

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
