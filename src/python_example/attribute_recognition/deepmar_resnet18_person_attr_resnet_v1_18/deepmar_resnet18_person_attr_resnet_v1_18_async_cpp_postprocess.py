#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
DeepMAR-ResNet18 Asynchronous Inference Example

C++ AttributePostProcess performs the sigmoid/softmax + threshold; convert_cpp_attribute attaches labels.

Usage:
    python deepmar_resnet18_person_attr_resnet_v1_18_async_cpp_postprocess.py --model model.dxnn --image input.jpg
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

from factory import Deepmar_resnet18_person_attr_resnet_v1_18Factory
from dx_postprocess import AttributePostProcess
from common.utility import convert_cpp_attribute
from common.processors.attribute_postprocessor import PETA_35_LABELS
from functools import partial
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("DeepMAR-ResNet18 Async Inference")
def main():
    args = parse_args()
    factory = Deepmar_resnet18_person_attr_resnet_v1_18Factory()

    def on_engine_init(runner):
        runner._cpp_postprocessor = AttributePostProcess(0.5, False)
        runner._cpp_convert_fn = partial(convert_cpp_attribute, labels=PETA_35_LABELS)

    runner = AsyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()
