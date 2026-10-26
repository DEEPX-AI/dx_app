"""tests/scripts/known_failures.py - TARGET's known tests/scripts failures as
strict xfails, applied only when --known-failures names a list (ci_checks.sh
does). Each case runs pytest on a throwaway suite under tmp_path."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import known_failures

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIST = HERE / "known_target_failures.txt"

SUITE = (
    "def test_fails():\n    assert 1 == 2\n\n"
    "def test_passes():\n    assert 1 == 1\n\n"
    "def test_other_failure():\n    assert 'a' == 'b'\n"
)


def _pytest(cwd, *args):
    env = {**os.environ, "PYTHONPATH": str(HERE), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "known_failures", "-p", "no:cacheprovider",
         "-q", "-rfExX", *args],
        cwd=cwd, capture_output=True, text=True, timeout=120, env=env)


def _suite(tmp_path, listing):
    (tmp_path / "test_demo.py").write_text(SUITE)
    (tmp_path / "known.txt").write_text(listing)
    return tmp_path


def test_a_listed_failure_is_an_expected_failure_with_its_reason(tmp_path):
    root = _suite(tmp_path, "test_demo.py::test_fails | red at the base commit\n")
    r = _pytest(root, "--known-failures", "known.txt", "test_demo.py::test_fails")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 xfailed" in r.stdout
    assert "red at the base commit" in r.stdout


def test_a_listed_test_that_passes_fails_the_run(tmp_path):
    # Strict: the day TARGET fixes a listed test, CI goes red until the entry is removed.
    root = _suite(tmp_path, "test_demo.py::test_passes | red at the base commit\n")
    r = _pytest(root, "--known-failures", "known.txt", "test_demo.py::test_passes")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "XPASS(strict)" in r.stdout
    assert "1 failed" in r.stdout


def test_an_unlisted_failure_still_fails(tmp_path):
    root = _suite(tmp_path, "test_demo.py::test_fails | red at the base commit\n")
    r = _pytest(root, "--known-failures", "known.txt")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAILED test_demo.py::test_other_failure" in r.stdout
    assert "1 failed" in r.stdout and "1 xfailed" in r.stdout


def test_without_the_option_a_listed_failure_is_reported_as_it_is(tmp_path):
    root = _suite(tmp_path, "test_demo.py::test_fails | red at the base commit\n")
    r = _pytest(root, "test_demo.py::test_fails")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAILED test_demo.py::test_fails" in r.stdout


@pytest.mark.parametrize("line", [
    "test_demo.py::test_fails\n",
    "test_demo.py::test_fails | \n",
    " | no test id\n",
    "test_demo.py | no test name\n",
])
def test_an_entry_without_a_test_id_and_a_reason_is_a_usage_error(tmp_path, line):
    root = _suite(tmp_path, "# comment\n\n" + line)
    r = _pytest(root, "--known-failures", "known.txt")
    assert r.returncode == 4, r.stdout + r.stderr
    assert "known.txt:3" in r.stderr


def test_a_test_listed_twice_is_a_usage_error(tmp_path):
    root = _suite(tmp_path, "test_demo.py::test_fails | one\ntest_demo.py::test_fails | two\n")
    r = _pytest(root, "--known-failures", "known.txt")
    assert r.returncode == 4, r.stdout + r.stderr
    assert "listed twice" in r.stderr


def test_a_missing_list_is_a_usage_error(tmp_path):
    root = _suite(tmp_path, "")
    r = _pytest(root, "--known-failures", "no-such-list.txt")
    assert r.returncode == 4, r.stdout + r.stderr


def test_ci_checks_runs_tests_scripts_with_the_list(tmp_path):
    # A copy of ci_checks.sh over a one-test tests/scripts whose conftest wires
    # the hooks the way the real one does: the listed failure keeps the check green.
    repo = tmp_path / "repo"
    suite = repo / "tests" / "scripts"
    suite.mkdir(parents=True)
    (repo / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "ci_checks.sh", repo / "scripts" / "ci_checks.sh")
    shutil.copy(HERE / "known_failures.py", suite / "known_failures.py")
    (suite / "conftest.py").write_text(
        "from known_failures import (  # noqa: F401\n"
        "    pytest_addoption, pytest_collection_modifyitems, pytest_configure)\n")
    (suite / "test_demo.py").write_text(SUITE.split("def test_passes")[0])
    (suite / "known_target_failures.txt").write_text("test_demo.py::test_fails | red at base\n")
    env = {k: v for k, v in os.environ.items() if k != "DXAPP_CI_CHECKS_ACTIVE"}
    env.update(PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(["bash", str(repo / "scripts" / "ci_checks.sh"), "--only", "tests-scripts"],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 xfailed" in r.stdout
    assert "PASS  tests-scripts" in r.stdout
    (suite / "known_target_failures.txt").write_text("# nothing known\n")
    red = subprocess.run(["bash", str(repo / "scripts" / "ci_checks.sh"), "--only", "tests-scripts"],
                         capture_output=True, text=True, timeout=300, env=env)
    assert red.returncode == 1, red.stdout + red.stderr
    assert "FAIL  tests-scripts" in red.stdout


def test_the_real_conftest_wires_the_hooks():
    tree = ast.parse((HERE / "conftest.py").read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "known_failures" for alias in node.names}
    assert imported == {"pytest_addoption", "pytest_configure", "pytest_collection_modifyitems"}


def test_an_empty_or_header_only_list_is_accepted(tmp_path):
    # The real list may hold no entry at all (only its header) once every
    # TARGET failure is fixed: ci_checks.sh keeps passing it, so it must load.
    for listing in ("", "# header only\n#\n# <test id> | <reason>\n\n"):
        root = _suite(tmp_path, listing)
        assert known_failures.load(root / "known.txt") == {}
        r = _pytest(root, "--known-failures", "known.txt", "test_demo.py::test_passes")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "1 passed" in r.stdout


def test_every_known_target_failure_names_a_test_that_exists():
    entries = known_failures.load(LIST)
    for test_id in entries:
        path, name = test_id.split("::", 1)
        source = (HERE / path).read_text(encoding="utf-8")
        func = name.split("[", 1)[0].split("::")[-1]
        assert "def {}(".format(func) in source, "{} is listed but not defined".format(test_id)
