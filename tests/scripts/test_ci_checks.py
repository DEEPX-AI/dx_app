"""scripts/ci_checks.sh - the one entry point CI and developers share (U-16).
Every case here uses --list or --only on a check that runs nothing heavy,
and never the tests-scripts check (it would run this suite again)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci_checks.sh"

EXPECTED = [
    "guard-graph-boundary", "guard-factory-uniqueness", "guard-model-registry",
    "guard-variant-scope", "codegen-strict", "codegen-check-docs", "header-odr", "cxx14-headers",
    "cross-compile", "python-compile", "workflow-yaml", "tests-scripts",
]


def _run(*args, script=SCRIPT, env=None):
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          timeout=300, env={**os.environ, "PYTHON": sys.executable, **(env or {})})


def _skeleton(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / "ci_checks.sh")
    for rel in ("src/python_example", "src/bindings/python", "tests", ".github/workflows"):
        (root / rel).mkdir(parents=True)
    return root


def _python_with_yaml():
    for py in (sys.executable, "python3", "/usr/bin/python3"):
        try:
            if subprocess.run([py, "-c", "import yaml"], capture_output=True).returncode == 0:
                return py
        except OSError:
            continue
    return None


def test_list_names_every_check_in_order():
    r = _run("--list")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == EXPECTED


def test_an_unknown_check_name_is_a_usage_error():
    r = _run("--only", "no-such-check")
    assert r.returncode == 2
    assert "unknown check" in r.stderr


def test_a_syntax_error_in_an_example_fails_python_compile(tmp_path):
    root = _skeleton(tmp_path)
    (root / "src" / "python_example" / "good.py").write_text("x = 1\n")
    ok = _run("--only", "python-compile", script=root / "scripts" / "ci_checks.sh")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "PASS  python-compile" in ok.stdout
    (root / "src" / "python_example" / "bad.py").write_text("def broken(:\n")
    bad = _run("--only", "python-compile", script=root / "scripts" / "ci_checks.sh")
    assert bad.returncode == 1
    assert "FAIL  python-compile" in bad.stdout
    assert not list(root.rglob("__pycache__")), "byte-compiled into the tree"


@pytest.mark.skipif(_python_with_yaml() is None, reason="no Python with PyYAML")
def test_a_broken_workflow_fails_workflow_yaml(tmp_path):
    root = _skeleton(tmp_path)
    workflows = root / ".github" / "workflows"
    (workflows / "good.yml").write_text(
        "name: x\non: push\njobs:\n  a:\n    runs-on: ubuntu-24.04\n    steps:\n      - run: 'true'\n")
    ok = _run("--only", "workflow-yaml", script=root / "scripts" / "ci_checks.sh")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    (workflows / "bad.yml").write_text("jobs: [\n")
    bad = _run("--only", "workflow-yaml", script=root / "scripts" / "ci_checks.sh")
    assert bad.returncode == 1
    assert "FAIL  workflow-yaml" in bad.stdout
    (workflows / "bad.yml").unlink()
    # Valid YAML, but no jobs: mapping - not a workflow GitHub would run.
    (workflows / "nojobs.yml").write_text("name: x\non: push\n")
    nojobs = _run("--only", "workflow-yaml", script=root / "scripts" / "ci_checks.sh")
    assert nojobs.returncode == 1, nojobs.stdout + nojobs.stderr
    assert "no jobs: mapping" in nojobs.stdout
    assert "FAIL  workflow-yaml" in nojobs.stdout


def test_a_check_that_itself_exits_77_fails_rather_than_skips(tmp_path):
    # 77 means SKIP only when ci_checks.sh itself gave a reason; a check
    # command that happens to exit 77 is a failure, not a silent pass.
    root = _skeleton(tmp_path)
    (root / "scripts" / "check_graph_boundary.py").write_text("import sys\nsys.exit(77)\n")
    r = _run("--only", "guard-graph-boundary", script=root / "scripts" / "ci_checks.sh")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAIL  guard-graph-boundary (exit 77)" in r.stdout
    assert "SKIP" not in r.stdout


def test_missing_dxrt_headers_skip_or_with_require_dxrt_fail(tmp_path):
    env = {"DXRT_INCLUDE_DIR": str(tmp_path / "no-dxrt")}
    r = _run("--only", "header-odr,cxx14-headers,cross-compile", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SKIP  header-odr: no dxrt headers at" in r.stdout
    assert "SKIP  cxx14-headers: no dxrt headers at" in r.stdout
    assert "SKIP  cross-compile:" in r.stdout
    strict = _run("--require-dxrt", "--only", "header-odr", env=env)
    assert strict.returncode == 1
    assert "FAIL  header-odr" in strict.stdout


def test_tests_scripts_is_never_started_from_inside_itself():
    r = _run("--only", "tests-scripts", env={"DXAPP_CI_CHECKS_ACTIVE": "1"})
    assert r.returncode == 1
    assert "already running inside ci_checks.sh" in r.stderr


def test_the_strict_claim_names_ci_checks():
    cmake = (ROOT / "src" / "cpp_example" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "catches in CI." not in cmake
    assert "codegen-strict check of scripts/ci_checks.sh" in cmake
    gen = " ".join((ROOT / "scripts" / "gen_model_registry.py").read_text(encoding="utf-8").split())
    assert "--strict (CI)" not in gen and 'drift (CI)"' not in gen
    assert gen.count("scripts/ci_checks.sh: codegen-strict") == 2  # docstring (re-wrapped) and --help
