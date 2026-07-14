#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Vit l 14 quickgelu dfn2b Synchronous Inference Example

Usage:
    python vit_l_14_quickgelu_dfn2b_sync.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Vit_l_14_quickgelu_dfn2bFactory
from common.runner import SyncRunner, parse_common_args


def parse_args():
    return parse_common_args("Vit l 14 quickgelu dfn2b Sync Inference", include_stream_inputs=False)


def main():
    args = parse_args()
    factory = Vit_l_14_quickgelu_dfn2bFactory()
    runner = SyncRunner(factory)
    runner.run(args)


if __name__ == "__main__":
    main()
