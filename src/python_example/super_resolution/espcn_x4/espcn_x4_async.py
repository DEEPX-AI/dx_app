#!/usr/bin/env python3
"""
Espcn_x4 Asynchronous Inference Example
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Espcn_x4Factory
from common.runner import AsyncRunner, parse_common_args

def parse_args():
    return parse_common_args("ESPCN-x4 Async Inference")
def main():
    args = parse_args()
    factory = Espcn_x4Factory()
    runner = AsyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
