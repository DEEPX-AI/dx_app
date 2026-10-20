#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""efficientnet-b2_288x288 sync_cpp_postprocess inference.

Post-processing runs in C++ through the dx_postprocess pybind11 module
(``ClassificationPostProcess``) instead of the Python postprocessor that
``efficientnet-b2_288x288_sync.py`` uses.

The variant is fixed to this folder. Run it from here with no PYTHONPATH::

    python efficientnet-b2_288x288_sync_cpp_postprocess.py -i <image>
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

from common.utility import convert_cpp_classification
from factory import EfficientnetFactory
from common.runner import SyncRunner

# Directory name is the .dxnn stem this script always builds.
_FIXED_VARIANT = "efficientnet-b2_288x288"


def on_engine_init(runner):
    """Swap the Python postprocessor for the C++ one from dx_postprocess.

    Imported here rather than at module level so ``--help`` works before
    the pybind11 module is built (``./build.sh``).
    """
    from dx_postprocess import ClassificationPostProcess

    config = runner.factory.config
    runner._cpp_postprocessor = ClassificationPostProcess(config.get("top_k", 5))
    runner._cpp_convert_fn = convert_cpp_classification


def main():
    run_entry(
        _script,
        factory_cls=EfficientnetFactory,
        runner_cls=SyncRunner,
        description="efficientnet-b2_288x288 sync_cpp_postprocess inference",
        fixed_variant=_FIXED_VARIANT,
        on_engine_init=on_engine_init,
    )


if __name__ == "__main__":
    main()
