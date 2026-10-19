#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""yolox-x_640x640 sync inference.

The variant is fixed to this folder. Run it from here with no PYTHONPATH::

    python yolox-x_640x640_sync.py -i <image>
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

from factory import YoloxFactory
from common.runner import SyncRunner

# Directory name is the .dxnn stem this script always builds.
_FIXED_VARIANT = "yolox-x_640x640"


def main():
    run_entry(
        _script,
        factory_cls=YoloxFactory,
        runner_cls=SyncRunner,
        description="yolox-x_640x640 sync inference",
        fixed_variant=_FIXED_VARIANT,
    )


if __name__ == "__main__":
    main()
