"""Known TARGET failures in tests/scripts, run as strict xfails.

scripts/ci_checks.sh (tests-scripts) runs this suite with
``--known-failures tests/scripts/known_target_failures.txt``. Every test named
there is marked ``xfail(strict=True)``: its failure does not fail the run, and
the day it passes the run fails (XPASS(strict)) until its entry is removed. A
plain ``pytest tests/scripts`` reads no list and reports every failure as it is.

List format: one test per line, ``<test id> | <reason>``. The test id is
relative to the list file's directory (``test_x.py::test_name[param]``). Lines
starting with ``#`` and blank lines are ignored. An entry without a test id or
a reason, a test listed twice, or a missing list is a usage error (exit 4).

conftest.py imports the three hooks below; they are also a plugin on their
own (``pytest -p known_failures``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pytest

SEPARATOR = "|"
_KNOWN = pytest.StashKey[Tuple[Path, Dict[str, str]]]()


def load(path) -> Dict[str, str]:
    """``{test id: reason}``; ValueError naming the line of a bad entry."""
    entries: Dict[str, str] = {}
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        test_id, sep, reason = (part.strip() for part in raw.partition(SEPARATOR))
        if not sep or "::" not in test_id or not test_id.split("::", 1)[1] or not reason:
            raise ValueError("{}:{}: expected '<test file>::<test name> | <reason>', got {!r}"
                             .format(path, number, raw))
        if test_id in entries:
            raise ValueError("{}:{}: {} is listed twice".format(path, number, test_id))
        entries[test_id] = reason
    return entries


def pytest_addoption(parser):
    parser.addoption(
        "--known-failures", metavar="FILE", default=None,
        help="mark the tests FILE lists ('<test id> | <reason>' per line) xfail(strict=True)")


def pytest_configure(config):
    path = config.getoption("known_failures", default=None)
    if not path:
        return
    try:
        entries = load(path)
    except (OSError, ValueError) as exc:
        raise pytest.UsageError("--known-failures: {}".format(exc)) from exc
    config.stash[_KNOWN] = (Path(path).resolve().parent, entries)


def pytest_collection_modifyitems(config, items):
    known = config.stash.get(_KNOWN, None)
    if known is None:
        return
    base, entries = known
    for item in items:
        try:
            rel = Path(item.path).resolve().relative_to(base).as_posix()
        except ValueError:
            continue
        reason = entries.get(rel + "::" + item.nodeid.split("::", 1)[1])
        if reason is not None:
            item.add_marker(pytest.mark.xfail(
                strict=True, reason="known TARGET failure: " + reason))
