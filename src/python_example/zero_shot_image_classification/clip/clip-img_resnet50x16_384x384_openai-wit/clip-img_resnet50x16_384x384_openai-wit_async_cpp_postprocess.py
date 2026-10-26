#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""clip-img_resnet50x16_384x384_openai-wit async_cpp_postprocess inference.

Post-processing runs in C++ through the dx_postprocess pybind11 module
(``EmbeddingPostProcess``) instead of the Python postprocessor that
``clip-img_resnet50x16_384x384_openai-wit_async.py`` uses.

The variant is fixed to this folder. Run it from here with no PYTHONPATH::

    python clip-img_resnet50x16_384x384_openai-wit_async_cpp_postprocess.py -i <image>
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

from factory import ClipFactory
from common.runner import AsyncRunner

# Directory name is the .dxnn stem this script always builds.
_FIXED_VARIANT = "clip-img_resnet50x16_384x384_openai-wit"


def on_engine_init(runner):
    """Swap the Python postprocessor for the C++ one from dx_postprocess.

    Imported here rather than at module level so ``--help`` works before
    the pybind11 module is built (``./build.sh``).
    """
    from dx_postprocess import EmbeddingPostProcess

    runner._cpp_postprocessor = EmbeddingPostProcess()
    # Zero-shot labels come from prompt_bank.json, so that step stays in
    # Python and gets the C++ L2-normalised embedding instead of the raw
    # model output.
    runner._cpp_convert_fn = (
        lambda embedding, ctx: runner.postprocessor.process([embedding], ctx))


def main():
    run_entry(
        _script,
        factory_cls=ClipFactory,
        runner_cls=AsyncRunner,
        description="clip-img_resnet50x16_384x384_openai-wit async_cpp_postprocess inference",
        fixed_variant=_FIXED_VARIANT,
        on_engine_init=on_engine_init,
    )


if __name__ == "__main__":
    main()
