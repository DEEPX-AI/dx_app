#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""yolov5-s6_1280x1280 async inference.

The variant is fixed to this folder. Run it from here with no PYTHONPATH::

    python yolov5-s6_1280x1280_async.py -i <image>
"""
import sys
from pathlib import Path

_script = Path(__file__).resolve()
# Vendored ./common (extract layout) wins; otherwise walk up to src/python_example.
_script_dir = _script.parent
for _cursor in (_script_dir, *_script_dir.parents):
    if (_cursor / "common" / "runner" / "entry.py").is_file():
        if str(_cursor) not in sys.path:
            sys.path.insert(0, str(_cursor))
        break

from common.runner.entry import install_import_paths, run_entry

install_import_paths(_script)

from factory import Yolov5Factory
from common.runner import AsyncRunner

# Directory name is the .dxnn stem this script always builds.
_FIXED_VARIANT = "yolov5-s6_1280x1280"


def main():
    run_entry(
        _script,
        factory_cls=Yolov5Factory,
        runner_cls=AsyncRunner,
        description="yolov5-s6_1280x1280 async inference",
        fixed_variant=_FIXED_VARIANT,
    )


if __name__ == "__main__":
    main()
