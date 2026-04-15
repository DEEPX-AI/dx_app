#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
DeepMAR-ResNet18 Asynchronous Inference Example

NOTE: No C++ AttributePostProcess binding available.
      Uses Python attribute postprocessor from factory.

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
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("DeepMAR-ResNet18 Async Inference")
def main():
    args = parse_args()
    factory = Deepmar_resnet18_person_attr_resnet_v1_18Factory()

    runner = AsyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
