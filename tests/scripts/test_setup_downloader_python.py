"""setup_sample_models.sh picks the Python that runs scripts/download_models.py
(U-76): $DXAPP_SETUP_PYTHON, else the active virtualenv, else the dx-runtime
venv beside dx_app, else python3 on PATH. A conda python3 (3.13+) verifies
TLS in X.509 strict mode, which a TLS-inspecting proxy's chain can fail.

Every interpreter here is a stub that records its calls, and the wrapper
runs from a copy in tmp_path: nothing is downloaded, nothing in the
repository is touched."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

STUB = """#!/bin/bash
echo "$0 $*" >> "{log}"
[ "$1" = "-c" ] && echo "Python 0.0-stub, OpenSSL stub"
exit 0
"""


def _stub(path: Path, log: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STUB.format(log=log))
    path.chmod(0o755)
    return path


@pytest.fixture()
def world(tmp_path):
    """<tmp>/runtime/dx_app holds the wrapper; <tmp>/runtime/venv-dx-runtime
    stands where dx-runtime/install.sh puts its venv."""
    app = tmp_path / "runtime" / "dx_app"
    (app / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "setup_sample_models.sh", app / "setup_sample_models.sh")
    for f in ("color_env.sh", "common_util.sh"):
        shutil.copy2(ROOT / "scripts" / f, app / "scripts" / f)
    (app / "scripts" / "download_models.py").write_text("# never run: every python is a stub\n")
    log = tmp_path / "calls.log"
    pythons = {
        "explicit": _stub(tmp_path / "explicit" / "bin" / "python", log),
        "active": _stub(tmp_path / "active_venv" / "bin" / "python", log),
        "runtime": _stub(tmp_path / "runtime" / "venv-dx-runtime" / "bin" / "python", log),
        "path": _stub(tmp_path / "fakebin" / "python3", log),
    }
    return app, log, pythons


def _run(world, tmp_path, **env):
    app, _, _ = world
    base = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "DXAPP_SETUP_PYTHON")}
    base["PATH"] = "{}:{}".format(tmp_path / "fakebin", base.get("PATH", "/usr/bin:/bin"))
    return subprocess.run(
        ["bash", str(app / "setup_sample_models.sh"), "--output={}".format(tmp_path / "models"),
         "--dry-run"],
        cwd=str(app), env={**base, **env}, capture_output=True, text=True, timeout=60)


def _callers(log: Path):
    """The interpreter of every recorded call, resolved (SCRIPT_DIR is a realpath)."""
    if not log.exists():
        return []
    return [os.path.realpath(line.split(" ", 1)[0]) for line in log.read_text().splitlines()]


def _downloader_ran_with(log: Path):
    return [os.path.realpath(line.split(" ", 1)[0]) for line in
            (log.read_text().splitlines() if log.exists() else []) if "download_models.py" in line]


def test_an_explicit_python_wins(world, tmp_path):
    _, log, py = world
    r = _run(world, tmp_path, DXAPP_SETUP_PYTHON=str(py["explicit"]),
             VIRTUAL_ENV=str(tmp_path / "active_venv"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert _downloader_ran_with(log) == [os.path.realpath(py["explicit"])]
    assert set(_callers(log)) == {os.path.realpath(py["explicit"])}  # the requests check too


def test_a_missing_explicit_python_is_an_error(world, tmp_path):
    _, log, _ = world
    r = _run(world, tmp_path, DXAPP_SETUP_PYTHON=str(tmp_path / "no" / "python"))
    assert r.returncode == 1
    assert "DXAPP_SETUP_PYTHON" in r.stderr and "is not an executable Python" in r.stderr
    assert _downloader_ran_with(log) == []


@pytest.mark.parametrize("kind", ["directory", "builtin"])
def test_a_directory_or_a_shell_builtin_is_not_a_python(world, tmp_path, kind):
    _, log, py = world
    if kind == "directory":
        value = str(py["explicit"].parent)  # [ -x ] is true for a directory
    else:
        if shutil.which("shopt"):
            pytest.skip("this system has a shopt executable on PATH")
        value = "shopt"  # a bash builtin: `command -v` finds it, `type -P` does not
    r = _run(world, tmp_path, DXAPP_SETUP_PYTHON=value)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "DXAPP_SETUP_PYTHON={} is not an executable Python".format(value) in r.stderr
    assert _downloader_ran_with(log) == []


def test_an_explicit_command_name_is_looked_up_on_path(world, tmp_path):
    _, log, py = world
    r = _run(world, tmp_path, DXAPP_SETUP_PYTHON="python3")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _downloader_ran_with(log) == [os.path.realpath(py["path"])]


def test_the_active_virtualenv_comes_next(world, tmp_path):
    _, log, py = world
    r = _run(world, tmp_path, VIRTUAL_ENV=str(tmp_path / "active_venv"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert _downloader_ran_with(log) == [os.path.realpath(py["active"])]


def test_then_the_dx_runtime_venv(world, tmp_path):
    _, log, py = world
    r = _run(world, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _downloader_ran_with(log) == [os.path.realpath(py["runtime"])]


def test_then_python3_on_path(world, tmp_path):
    _, log, py = world
    shutil.rmtree(tmp_path / "runtime" / "venv-dx-runtime")
    r = _run(world, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _downloader_ran_with(log) == [os.path.realpath(py["path"])]


def test_the_chosen_python_is_named(world, tmp_path):
    _, _, py = world
    r = _run(world, tmp_path, DXAPP_SETUP_PYTHON=str(py["explicit"]))
    assert "[DXAPP] [INFO] Downloader Python: {}".format(py["explicit"]) in r.stdout, r.stdout
    assert "OpenSSL stub" in r.stdout


def test_setup_help_names_the_download_environment():
    text = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert "DXAPP_SETUP_PYTHON=<python>" in text
    assert "DXAPP_TLS_RELAX_X509_STRICT=1" in text
