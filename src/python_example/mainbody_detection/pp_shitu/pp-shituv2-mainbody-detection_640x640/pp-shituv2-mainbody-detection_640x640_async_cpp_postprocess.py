#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""pp-shituv2-mainbody-detection_640x640 async_cpp_postprocess inference.

Post-processing runs in C++ through the dx_postprocess pybind11 module
(``PicoDetPostProcess``) instead of the Python postprocessor that
``pp-shituv2-mainbody-detection_640x640_async.py`` uses.

The variant is fixed to this folder. Run it from here with no PYTHONPATH::

    python pp-shituv2-mainbody-detection_640x640_async_cpp_postprocess.py -i <image>
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

from common.utility import convert_cpp_detections
from factory import PpShituFactory
from common.runner import AsyncRunner

# Directory name is the .dxnn stem this script always builds.
_FIXED_VARIANT = "pp-shituv2-mainbody-detection_640x640"


def on_engine_init(runner):
    """Swap the Python postprocessor for the C++ one from dx_postprocess.

    Imported here rather than at module level so ``--help`` works before
    the pybind11 module is built (``./build.sh``).
    """
    from dx_postprocess import PicoDetPostProcess

    config = runner.factory.config
    runner._cpp_postprocessor = PicoDetPostProcess(
        runner.input_width, runner.input_height,
        float(config.get("conf_threshold", config.get("score_threshold", 0.3))),
        float(config.get("nms_threshold", 0.5)),
        int(config.get("num_classes", 1)))
    runner._cpp_convert_fn = convert_cpp_detections


def main():
    run_entry(
        _script,
        factory_cls=PpShituFactory,
        runner_cls=AsyncRunner,
        description="pp-shituv2-mainbody-detection_640x640 async_cpp_postprocess inference",
        fixed_variant=_FIXED_VARIANT,
        on_engine_init=on_engine_init,
    )


if __name__ == "__main__":
    main()
