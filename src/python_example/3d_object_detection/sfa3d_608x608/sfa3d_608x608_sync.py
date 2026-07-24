#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
SFA3D Synchronous Inference Example

Usage:
    python sfa3d_608x608_sync.py --model model.dxnn --image pointcloud.bin
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from factory import Sfa3d608x608Factory
from common.runner import SyncRunner, parse_common_args
from common.utility.kitti_calib import apply_kitti_companion_dirs_from_args
from calib_policy import enforce_calib_policy


def parse_args():
    return parse_common_args(
        "SFA3D 608x608 Sync Inference",
        include_kitti_paths=True,
    )


def main():
    args = parse_args()
    apply_kitti_companion_dirs_from_args(args)
    enforce_calib_policy(args)
    factory = Sfa3d608x608Factory()
    runner = SyncRunner(factory)
    runner.run(args)


if __name__ == "__main__":
    main()
