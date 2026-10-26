"""The C++ and Python suites register their shared options once per session.

Both ``tests/cpp_example/conftest.py`` and ``tests/python_example/conftest.py``
register ``--loop``, ``--camera-index``, ``--rtsp-url`` and
``--stream-duration`` through ``test_helpers.pytest_support``. A session that
loads both conftests used to abort before collecting anything with
``ValueError: option names {'--camera-index'} already added``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest.config.argparsing import Parser

TESTS_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TESTS_DIR.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_helpers.pytest_support import add_loop_option, add_stream_options  # noqa: E402

SHARED = ("--loop", "--camera-index", "--rtsp-url", "--stream-duration")


def _option_names(parser: Parser) -> list:
    return [name for option in parser._anonymous.options for name in option.names()]


def test_registering_the_shared_options_twice_keeps_one_of_each():
    parser = Parser()
    for default in ("20", "1"):          # C++ image conftest, then the Python one
        add_loop_option(parser, default=default)
        add_stream_options(parser)
    names = _option_names(parser)
    for option in SHARED:
        assert names.count(option) == 1, (option, names)
    loop = next(o for o in parser._anonymous.options if "--loop" in o.names())
    assert loop.default == "20", "the first registration must win"


def test_an_option_error_other_than_a_duplicate_still_raises():
    from test_helpers.pytest_support import _addoption_once

    parser = Parser()
    _addoption_once(parser, "--loop", default="1")
    with pytest.raises(Exception) as excinfo:
        _addoption_once(parser, "--x", action="no-such-action")
    assert "already added" not in str(excinfo.value)


def test_one_session_loads_both_suites_conftests():
    """``--help`` runs every initial conftest's ``pytest_addoption`` and stops."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
         "tests/cpp_example", "tests/python_example", "--help"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300, env=env)
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]
    assert "already added" not in out, out[-3000:]
    for option in SHARED + ("--coverage",):
        assert option in result.stdout, option
