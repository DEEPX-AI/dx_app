"""The hermetic guard sees what U-74 did - on a fake tree, never the real one."""
from __future__ import annotations

import os
import shutil

from pathlib import Path

import pytest

import hermetic_guard as guard

pytest_plugins = ["pytester"]

HERE = Path(__file__).resolve().parent

# Every fake entry starts at a fixed old time, so any later write is newer
# even within one filesystem timestamp tick.
OLD_NS = 1_000_000_000_000_000_000

TRACKED = ("build.sh", "src/cpp_example/classification/alexnet/alexnet_sync.cpp")


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "repo"
    for rel in ("build_x86_64/CMakeCache.txt", "build_x86_64/release/bin/yolov7_sync",
                "bin/yolov7_sync", "bin/python/dx_graph/_dx_graph.so",
                "src/cpp_example/classification/alexnet/alexnet_sync.cpp",
                "build.sh", ".cache/scratch.txt"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            os.utime(os.path.join(dirpath, name), ns=(OLD_NS, OLD_NS))
    return root


def changes(root, mutate):
    before = guard.snapshot(root, TRACKED)
    mutate(root)
    return guard.diff(before, guard.snapshot(root, TRACKED))


def test_an_untouched_tree_has_no_changes(tree):
    assert changes(tree, lambda r: None) == []


def test_deleting_the_release_tree_is_seen(tree):
    # build.sh: rm -rf $build_dir/release, on every configure.
    found = changes(tree, lambda r: shutil.rmtree(r / "build_x86_64" / "release"))
    assert "removed build_x86_64/release" in found
    assert "changed build_x86_64" in found


def test_a_reconfigure_rewriting_the_cache_is_seen(tree):
    found = changes(tree, lambda r: (r / "build_x86_64" / "CMakeCache.txt").write_text("y"))
    assert found == ["changed build_x86_64/CMakeCache.txt"]


def test_a_new_build_directory_is_seen(tree):
    assert changes(tree, lambda r: (r / "build_aarch64").mkdir()) == ["added build_aarch64"]


def test_a_deep_install_file_is_seen(tree):
    found = changes(tree, lambda r: (r / "bin/python/dx_graph/_dx_graph.so").write_text("x"))
    assert found == ["changed bin/python/dx_graph/_dx_graph.so"]


def test_a_tracked_file_rewritten_with_its_own_bytes_is_seen(tree):
    def rewrite(r):
        path = r / "build.sh"
        path.write_text(path.read_text())
    assert changes(tree, rewrite) == ["changed build.sh"]


def test_a_category_directory_made_for_a_test_is_seen(tree):
    found = changes(tree, lambda r: (r / "src/cpp_example/dxapp_empty_test_category").mkdir())
    assert found == ["changed ls:src/cpp_example"]


def test_untracked_scratch_outside_the_protected_trees_is_allowed(tree):
    assert changes(tree, lambda r: (r / ".cache" / "scratch.txt").write_text("y")) == []


def test_the_guard_runs_around_every_test(request):
    assert "_real_tree_untouched" in request.fixturenames


def test_an_entry_gone_before_its_lstat_is_not_an_error(tmp_path):
    # scandir listed it, then it went: the walk skips it and diff() reports
    # "removed", instead of a FileNotFoundError out of the fixture.
    out = {}
    guard._walk(str(tmp_path / "gone"), "bin/gone", out)
    assert out == {}


def test_a_git_checkout_listing_no_tracked_file_is_a_problem(tmp_path):
    assert guard.tracked_list_problem(tmp_path, []) is None  # not a checkout: nothing to list
    (tmp_path / ".git").write_text("gitdir: /elsewhere\n")  # a worktree's .git is a file
    assert "git ls-files listed no tracked file" in guard.tracked_list_problem(tmp_path, [])
    assert guard.tracked_list_problem(tmp_path, ["build.sh"]) is None


def _fake_repo(pytester):
    """pytester's directory as a repository root: this suite's conftest and
    the modules it imports under tests/scripts/, and a bin/ for a test to dirty."""
    scripts = pytester.path / "tests" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("conftest.py", "hermetic_guard.py", "known_failures.py"):
        shutil.copy(HERE / name, scripts / name)
    (pytester.path / "bin").mkdir()
    (pytester.path / "bin" / "x").write_text("x")
    return scripts


def test_a_test_that_dirties_the_tree_fails(pytester):
    scripts = _fake_repo(pytester)
    (scripts / "test_dirty.py").write_text(
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[2]\n"
        "def test_clean():\n    pass\n"
        "def test_dirty():\n    (ROOT / 'bin' / 'x').write_text('changed')\n")
    result = pytester.runpytest_subprocess(str(scripts), "-p", "no:cacheprovider")
    out = result.stdout.str()
    assert result.parseoutcomes().get("errors") == 1, out
    assert "test_dirty.py::test_dirty changed the real repository" in out, out
    assert "changed bin/x" in out, out
    assert "test_clean changed" not in out, out


def test_a_checkout_git_cannot_list_stops_the_run(pytester):
    # CI's "dubious ownership": git fails, the tracked-file half of the guard
    # would be off. The run stops before any test instead.
    scripts = _fake_repo(pytester)
    (pytester.path / ".git").write_text("gitdir: /nonexistent\n")
    (scripts / "test_ok.py").write_text("def test_ok():\n    pass\n")
    result = pytester.runpytest_subprocess(str(scripts), "-p", "no:cacheprovider")
    assert result.ret == 4, result.stdout.str() + result.stderr.str()
    assert "git ls-files listed no tracked file" in result.stdout.str() + result.stderr.str()
