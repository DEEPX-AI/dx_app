"""A C++ example started without a display server and without --no-display.

With no DISPLAY / WAYLAND_DISPLAY the first HighGUI window is fatal: Qt's xcb
plugin calls qFatal() (SIGABRT), and the GTK backend of OpenCV < 4.5 (apt
OpenCV 4.2 on Ubuntu 20.04) calls exit(1) from gtk_init(). Neither can be
caught, so the runner has to check hasDisplay() and run as --no-display.

setup_environment() sets QT_QPA_PLATFORM=offscreen when there is no display,
which hides exactly this, so the test removes it again.
"""
import os
import subprocess

import pytest

# conftest.py puts tests/ on sys.path; hence the noqa: E402 imports below.
from test_helpers.constants import MODELS_DIR, PROJECT_ROOT  # noqa: E402
from test_helpers.utils import binary_path, setup_environment  # noqa: E402

from conftest import resolve_bin_dir

BIN_DIR = resolve_bin_dir()
# One embedding example stands for the shared sync/async embedding runners.
EMBEDDING_MODEL = "eigenplaces-resnet18_512x512"
IMAGE = PROJECT_ROOT / "sample" / "img" / "sample_dog.jpg"


@pytest.mark.e2e
@pytest.mark.e2e_image
@pytest.mark.skipif(os.name == "nt", reason="Windows always has a native window system")
@pytest.mark.parametrize("variant", ["sync", "async"])
def test_embedding_example_without_a_display_runs_headless(variant):
    exe = binary_path(BIN_DIR, "{}_{}".format(EMBEDDING_MODEL, variant))
    if not exe.exists():
        pytest.skip("Binary not found: {}".format(exe.name))
    model = MODELS_DIR / (EMBEDDING_MODEL + ".dxnn")
    if not model.exists():
        pytest.skip("{} absent - run ./setup.sh --models {}".format(model.name, EMBEDDING_MODEL))

    env = setup_environment()
    for var in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        env.pop(var, None)
    result = subprocess.run(
        [str(exe), "-m", str(model), "-i", str(IMAGE), "-l", "1"],
        capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT), timeout=120)
    output = result.stdout + result.stderr
    assert result.returncode == 0, "rc={}:\n{}".format(result.returncode, output[-2000:])
    assert "PERFORMANCE SUMMARY" in output, output[-2000:]
