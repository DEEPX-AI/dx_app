"""Unit test: missing-image error message includes a sample-image hint.

Mirrors the friendly guidance already provided for missing model/video
paths in the runner's input validation.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add python_example to path so we can import common.* (matches conftest).
_src = str(Path(__file__).resolve().parent.parent.parent.parent / "src" / "python_example")
if _src not in sys.path:
    sys.path.insert(0, _src)

from common.runner.sync_runner import _validate_media  # noqa: E402


def test_missing_image_path_exits():
    """A non-existent --image path should abort with exit code 1."""
    args = SimpleNamespace(image="/nonexistent/does_not_exist.jpg", video=None)
    with pytest.raises(SystemExit) as exc:
        _validate_media(args)
    assert exc.value.code == 1


def test_missing_image_path_includes_sample_hint(caplog):
    """The error should point users to bundled sample images."""
    args = SimpleNamespace(image="/nonexistent/does_not_exist.jpg", video=None)
    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            _validate_media(args)
    assert "sample/img" in caplog.text
