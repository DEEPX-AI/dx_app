"""Common fixtures for postprocessor unit tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add python_example to path so we can import common.*
_src = str(Path(__file__).resolve().parent.parent.parent.parent / "src" / "python_example")
if _src not in sys.path:
    sys.path.insert(0, _src)

from common.base.i_processor import PreprocessContext  # noqa: E402


@pytest.fixture
def ctx():
    """Default preprocessing context (no letterbox, 1:1 scale)."""
    return PreprocessContext(
        original_width=640, original_height=480,
        input_width=640, input_height=640,
        scale=1.0,
    )


@pytest.fixture
def ctx_letterbox():
    """Preprocessing context with letterbox padding (typical detection model)."""
    return PreprocessContext(
        pad_x=0, pad_y=80,
        scale=1.0, scale_x=1.0, scale_y=1.0,
        original_width=640, original_height=480,
        input_width=640, input_height=640,
    )
