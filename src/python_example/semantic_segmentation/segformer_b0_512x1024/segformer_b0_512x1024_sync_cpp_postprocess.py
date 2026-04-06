#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Deeplabv3 Synchronous Inference Example

Usage:
    python segformer_b0_512x1024_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import DeepLabv3PostProcess
from dx_engine import InferenceOption
from common.utility.visualization import deeplabv3_cpp_visualize
from factory import Segformer_b0_512x1024Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("SegFormer-B0 Sync Inference")
def main():
    args = parse_args()
    factory = Segformer_b0_512x1024Factory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        runner._cpp_postprocessor = DeepLabv3PostProcess(input_w, input_h)
        runner._cpp_visualize_fn = deeplabv3_cpp_visualize

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
