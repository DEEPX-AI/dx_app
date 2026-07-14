#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Depth anything v2 vits Synchronous Inference Example

Usage:
    python depth_anything_v2_vits_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
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


from dx_postprocess import DepthPostProcess
from common.utility.visualization import depth_cpp_visualize
from factory import Depth_anything_v2_vitsFactory
from common.runner import SyncRunner, parse_common_args


def parse_args():
    return parse_common_args("Depth anything v2 vits Sync Inference")


def main():
    args = parse_args()
    factory = Depth_anything_v2_vitsFactory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = DepthPostProcess(runner.input_width, runner.input_height)
        runner._cpp_visualize_fn = depth_cpp_visualize

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)


if __name__ == "__main__":
    main()
