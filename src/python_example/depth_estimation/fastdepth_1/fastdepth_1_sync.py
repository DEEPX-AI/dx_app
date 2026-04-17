#!/usr/bin/env python3
"""
Fastdepth_1 Synchronous Inference Example
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Fastdepth_1Factory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("FastDepth Sync Inference", include_output=True)
def main():
    args = parse_args()
    factory = Fastdepth_1Factory()
    runner = SyncRunner(factory)
    runner.run(args)

if __name__ == "__main__":
    main()
