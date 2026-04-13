#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
UNet-MobileNetV2 Synchronous Inference Example

NOTE: NHWC 3-class float logits — incompatible with C++ SemanticSegPostProcess input format (CHW).
      Falls back to Python postprocessing instead of C++ PostProcess binding.

Usage:
    python unet_mobilenet_v2_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Unet_mobilenet_v2Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("UNet-MobileNetV2 Synchronous Inference")
def main():
    args = parse_args()
    factory = Unet_mobilenet_v2Factory()

    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
