#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""yolo26-s-pose_640x640_pre-optimized sync_cpp_postprocess inference.

Post-processing runs in C++ through the dx_postprocess pybind11 module
(``PreoptTopKPostProcess``) instead of the Python postprocessor that
``yolo26-s-pose_640x640_pre-optimized_sync.py`` uses.

The variant is fixed to this folder. Run it from here with no PYTHONPATH::

    python yolo26-s-pose_640x640_pre-optimized_sync_cpp_postprocess.py -i <image>
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

from common.utility import convert_cpp_pose_detections
from factory import YoloPreoptPoseFactory
from common.runner import SyncRunner

# Directory name is the .dxnn stem this script always builds.
_FIXED_VARIANT = "yolo26-s-pose_640x640_pre-optimized"


def on_engine_init(runner):
    """Swap the Python postprocessor for the C++ one from dx_postprocess.

    Imported here rather than at module level so ``--help`` works before
    the pybind11 module is built (``./build.sh``).
    """
    from dx_postprocess import PreoptTopKPostProcess

    config = runner.factory.config
    runner._cpp_postprocessor = PreoptTopKPostProcess(
        runner.input_width, runner.input_height,
        float(config.get("score_threshold", config.get("conf_threshold", 0.3))),
        float(config.get("nms_threshold", 0.45)),
        int(config.get("top_k", 300)),
        int(config.get("num_keypoints", 17)))
    runner._cpp_convert_fn = convert_cpp_pose_detections


def main():
    run_entry(
        _script,
        factory_cls=YoloPreoptPoseFactory,
        runner_cls=SyncRunner,
        description="yolo26-s-pose_640x640_pre-optimized sync_cpp_postprocess inference",
        fixed_variant=_FIXED_VARIANT,
        on_engine_init=on_engine_init,
    )


if __name__ == "__main__":
    main()
