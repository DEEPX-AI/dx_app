#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
FastDepth Synchronous Inference Example

Usage:
    python fastdepth_1_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import DepthPostProcess
from common.utility.visualization import depth_cpp_visualize
from factory import Fastdepth_1Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("FastDepth-1 Sync Inference")
def main():
    args = parse_args()
    factory = Fastdepth_1Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        runner._cpp_postprocessor = DepthPostProcess(input_w, input_h)
        runner._cpp_visualize_fn = depth_cpp_visualize

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
