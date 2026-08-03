"""Unit test: missing-image path is rejected with an actionable error.

Per the SDKREQ-529 policy the runner never falls back to a bundled sample and
never auto-downloads, so the error only has to abort and name the path that
was not found.
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


def test_missing_image_path_error_names_the_offending_path(caplog):
    """The error must name the path that was not found, so users can fix the CLI arg."""
    args = SimpleNamespace(image="/nonexistent/does_not_exist.jpg", video=None)
    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            _validate_media(args)
    assert "not found" in caplog.text
    assert "/nonexistent/does_not_exist.jpg" in caplog.text
