#!/usr/bin/env python3
"""
SSDMV1 Asynchronous Inference Example
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Ssdmv2liteFactory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("SSDMV2Lite Async Inference")
def main():
    args = parse_args()
    factory = Ssdmv2liteFactory()
    runner = AsyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
