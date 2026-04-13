#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
EfficientNet Asynchronous Inference Example

Usage:
    python efficientnetb5_async_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import ClassificationPostProcess
from common.utility import convert_cpp_classification
from factory import Efficientnetb5Factory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("EfficientNet-Lite0 Async Inference")
def main():
    args = parse_args()
    factory = Efficientnetb5Factory()

    def on_engine_init(runner):
        config = runner.factory.config
        runner._cpp_postprocessor = ClassificationPostProcess(config.get("top_k", 5))
        runner._cpp_convert_fn = convert_cpp_classification

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
