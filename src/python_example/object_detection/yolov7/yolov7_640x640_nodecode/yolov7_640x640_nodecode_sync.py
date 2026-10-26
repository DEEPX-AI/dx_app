#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""yolov7_640x640_nodecode sync inference.

The variant is fixed to this folder. Run it from here with no PYTHONPATH::

    python yolov7_640x640_nodecode_sync.py -i <image>
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

from factory import Yolov7Factory
from common.runner import SyncRunner

# Directory name is the .dxnn stem this script always builds.
_FIXED_VARIANT = "yolov7_640x640_nodecode"


def main():
    run_entry(
        _script,
        factory_cls=Yolov7Factory,
        runner_cls=SyncRunner,
        description="yolov7_640x640_nodecode sync inference",
        fixed_variant=_FIXED_VARIANT,
    )


if __name__ == "__main__":
    main()
