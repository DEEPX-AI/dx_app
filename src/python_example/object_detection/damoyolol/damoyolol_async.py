#!/usr/bin/env python3
"""
DamoYolo Asynchronous Inference Example - DX-APP v3.0.0
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import DamoyololFactory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("DAMO-YOLO-L Async Inference")
def main():
    args = parse_args()
    factory = DamoyololFactory()
    runner = AsyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
