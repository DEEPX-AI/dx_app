"""Tests for scripts/gui_env.sh (OpenCV HighGUI start-up warning helpers)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _run(snippet: str, env: dict) -> subprocess.CompletedProcess:
    script = (
        f'source "{SCRIPTS}/color_env.sh"\n'
        f'source "{SCRIPTS}/common_util.sh"\n'
        f'source "{SCRIPTS}/gui_env.sh"\n'
        f"{snippet}\n"
    )
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=False)


def _base_env(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "NO_AT_BRIDGE"}
    env["XDG_RUNTIME_DIR"] = str(tmp_path / "runtime")
    return env


@pytest.fixture
def fake_cv2(tmp_path: Path) -> Path:
    """A cv2 package shaped like the opencv-python wheel: qt/plugins, no qt/fonts."""
    pkg = tmp_path / "site" / "cv2"
    (pkg / "qt" / "plugins").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    return pkg


def _python_env(tmp_path: Path, fake_cv2: Path, font_dir: Path) -> dict:
    env = _base_env(tmp_path)
    env["PYTHONPATH"] = str(fake_cv2.parent)
    env["DXAPP_TEST_FONT_DIR"] = str(font_dir)
    return env


LINK_WITH_TEST_FONTS = (
    'DXAPP_GUI_FONT_DIRS=("$DXAPP_TEST_FONT_DIR"); '
    f'dxapp_link_cv2_qt_fonts "{sys.executable}"'
)


def test_no_at_bridge_set_when_bus_missing(tmp_path):
    result = _run('dxapp_prepare_gui_env; echo "[$NO_AT_BRIDGE]"', _base_env(tmp_path))
    assert result.stdout.strip() == "[1]"


def test_no_at_bridge_kept_when_user_set_it(tmp_path):
    env = _base_env(tmp_path)
    env["NO_AT_BRIDGE"] = "0"
    result = _run('dxapp_prepare_gui_env; echo "[$NO_AT_BRIDGE]"', env)
    assert result.stdout.strip() == "[0]"


def test_no_at_bridge_unset_when_bus_exists(tmp_path):
    import socket

    bus_dir = tmp_path / "runtime" / "at-spi"
    bus_dir.mkdir(parents=True)
    sock = socket.socket(socket.AF_UNIX)
    try:
        sock.bind(str(bus_dir / "bus"))
        result = _run('dxapp_prepare_gui_env; echo "[${NO_AT_BRIDGE-unset}]"', _base_env(tmp_path))
    finally:
        sock.close()
    assert result.stdout.strip() == "[unset]"


def test_links_missing_cv2_font_dir(tmp_path, fake_cv2):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    result = _run(LINK_WITH_TEST_FONTS, _python_env(tmp_path, fake_cv2, font_dir))
    assert result.returncode == 0, result.stderr
    link = fake_cv2 / "qt" / "fonts"
    assert link.is_symlink()
    assert link.resolve() == font_dir.resolve()


def test_existing_cv2_font_dir_untouched(tmp_path, fake_cv2):
    own_fonts = fake_cv2 / "qt" / "fonts"
    own_fonts.mkdir()
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    result = _run(LINK_WITH_TEST_FONTS, _python_env(tmp_path, fake_cv2, font_dir))
    assert result.returncode == 0, result.stderr
    assert own_fonts.is_dir() and not own_fonts.is_symlink()


def test_headless_cv2_skipped(tmp_path, fake_cv2):
    (fake_cv2 / "qt" / "plugins").rmdir()
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    result = _run(LINK_WITH_TEST_FONTS, _python_env(tmp_path, fake_cv2, font_dir))
    assert result.returncode == 0, result.stderr
    assert not (fake_cv2 / "qt" / "fonts").exists()


def test_no_system_fonts_warns_but_succeeds(tmp_path, fake_cv2):
    result = _run(LINK_WITH_TEST_FONTS, _python_env(tmp_path, fake_cv2, tmp_path / "missing"))
    assert result.returncode == 0
    assert "fonts-dejavu-core" in result.stderr
    assert not (fake_cv2 / "qt" / "fonts").exists()


def test_missing_python_is_noop(tmp_path):
    result = _run("dxapp_link_cv2_qt_fonts /nonexistent/python3", _base_env(tmp_path))
    assert result.returncode == 0
    assert result.stderr == ""
