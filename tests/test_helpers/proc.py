"""
Subprocess helpers shared by the C++ and Python example suites.

* :func:`example_python` - the interpreter that runs the Python examples.
* :func:`apply_headless_display` - Qt ``offscreen`` when there is no display.
* :func:`run_bounded` - ``subprocess.run`` whose timeout sends SIGTERM first
  and SIGKILL only if the process ignores it.
* :func:`wait_until_blocked_opening_a_fifo` - poll ``/proc`` until a process
  sleeps in ``open()`` of a FIFO nobody has opened for reading.
"""
from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .constants import PROJECT_ROOT

# ======================================================================
# Interpreter for the Python examples
# ======================================================================

#: Environment variable naming the interpreter that runs the Python examples.
TEST_PYTHON_ENV = "DXAPP_TEST_PYTHON"

#: The dx-runtime venv (``dx-runtime/venv-dx-runtime``) that ``install.sh``
#: creates next to ``dx_app``; it has dx_engine, dx_postprocess and OpenCV.
DX_RUNTIME_VENV = PROJECT_ROOT.parent / "venv-dx-runtime"


def _venv_python(venv: Path) -> Optional[Path]:
    for rel in ("bin/python3", "bin/python", "Scripts/python.exe"):
        candidate = venv / rel
        if candidate.is_file():
            # Not resolved: bin/python is a symlink to the base interpreter,
            # and only the venv path gives the venv's site-packages.
            return candidate
    return None


def example_python() -> str:
    """Interpreter for every Python example subprocess.

    ``$DXAPP_TEST_PYTHON`` when set, else the dx-runtime venv python when that
    venv exists, else the interpreter running pytest. The examples import
    dx_engine / dx_postprocess / cv2 for real (the conftest mocks exist only
    in the pytest process), so running them under an interpreter without
    those modules fails every test with ModuleNotFoundError.
    """
    override = os.environ.get(TEST_PYTHON_ENV)
    if override:
        found = shutil.which(override)
        if found is None:
            raise RuntimeError(
                "{}={!r} is not an executable interpreter".format(TEST_PYTHON_ENV, override))
        return found
    venv = _venv_python(DX_RUNTIME_VENV)
    if venv is not None:
        return str(venv)
    return sys.executable


# ======================================================================
# Headless display
# ======================================================================

def apply_headless_display(env) -> None:
    """Without ``DISPLAY``/``WAYLAND_DISPLAY``, default ``QT_QPA_PLATFORM`` to
    ``offscreen`` in *env* (a dict or ``os.environ``).

    OpenCV's Qt backend aborts the process (rc 134) when it cannot reach an X
    server, so any C++ example that opens a window dies before it does any
    work. An explicit ``QT_QPA_PLATFORM`` is left alone.
    """
    if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        env.setdefault("QT_QPA_PLATFORM", "offscreen")


# ======================================================================
# Bounded subprocess runs
# ======================================================================

#: Seconds between the SIGTERM and the SIGKILL of a timed-out process.
DEFAULT_GRACE_S = 10.0


class BoundedTimeout(subprocess.TimeoutExpired):
    """A :func:`run_bounded` timeout.

    ``returncode`` is the exit status after the SIGTERM (or SIGKILL);
    ``killed`` says whether SIGKILL was needed, i.e. the process ignored
    SIGTERM for the whole grace period.
    """

    def __init__(self, cmd, timeout, output, stderr, returncode, killed):
        super().__init__(cmd, timeout, output=output, stderr=stderr)
        self.returncode = returncode
        self.killed = killed

    def __str__(self):
        how = "SIGKILL (ignored SIGTERM)" if self.killed else "SIGTERM"
        return "{} ended by {} after {}s timeout (rc={})".format(
            self.cmd, how, self.timeout, self.returncode)


def _signal_group(proc: subprocess.Popen, sig) -> None:
    if os.name == "nt":
        if sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_bounded(cmd, *, timeout=None, grace=DEFAULT_GRACE_S, input=None,
                capture_output=False, check=False, **popen_kwargs):
    """``subprocess.run`` with a timeout that escalates SIGTERM -> SIGKILL.

    The child gets its own process group (POSIX), so the signals reach
    anything it started. On timeout the group gets SIGTERM, then SIGKILL if it
    is still alive after *grace* seconds, and :class:`BoundedTimeout` (a
    ``subprocess.TimeoutExpired``) is raised with the output collected so
    far. ``subprocess.run`` itself goes straight to SIGKILL and leaves any
    grandchildren running.
    """
    if capture_output:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE
    if input is not None:
        popen_kwargs["stdin"] = subprocess.PIPE
    if os.name != "nt":
        popen_kwargs.setdefault("start_new_session", True)

    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        try:
            out, err = proc.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGTERM)
            killed = False
            try:
                out, err = proc.communicate(timeout=grace)
            except subprocess.TimeoutExpired:
                killed = True
                _signal_group(proc, signal.SIGKILL)
                try:
                    out, err = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    out, err = None, None  # pipe held by an escaped process
            raise BoundedTimeout(cmd, timeout, out, err, proc.poll(), killed)
    except BaseException:
        # KeyboardInterrupt etc.: the child is in its own session and would
        # not see the terminal's Ctrl-C, so never leave it behind.
        if proc.poll() is None:
            _signal_group(proc, signal.SIGKILL)
        raise
    finally:
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    result = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if check:
        result.check_returncode()
    return result


# ======================================================================
# A process blocked opening a FIFO
# ======================================================================

#: Where a thread sleeps while open() of a FIFO waits for the other end.
FIFO_OPEN_WCHANS = ("fifo_open", "wait_for_partner")

#: The openat syscall number, read only when the kernel hides wchan.
_OPENAT = {"x86_64": "257", "aarch64": "56"}


def _thread_blocked_opening_a_fifo(task: Path) -> bool:
    try:
        wchan = (task / "wchan").read_text().strip()
    except OSError:
        wchan = ""
    if wchan not in ("", "0"):
        return wchan in FIFO_OPEN_WCHANS
    openat = _OPENAT.get(platform.machine())
    try:
        fields = (task / "syscall").read_text().split()
    except OSError:
        return False
    return openat is not None and bool(fields) and fields[0] == openat


def blocked_opening_a_fifo(pid: int) -> bool:
    """True when a thread of *pid* sleeps in open() of a FIFO (Linux ``/proc``)."""
    return any(_thread_blocked_opening_a_fifo(task)
               for task in Path("/proc/{}/task".format(pid)).glob("*"))


def wait_until_blocked_opening_a_fifo(proc: subprocess.Popen, timeout: float = 60.0,
                                      output=lambda: "") -> None:
    """Return once *proc* is blocked opening a FIFO; fail with *output()* if
    it exits first or *timeout* seconds pass. A fixed sleep guessed that
    point, and under NPU contention a signal meant for the blocked open
    could land during inference instead."""
    deadline = time.monotonic() + timeout
    while not blocked_opening_a_fifo(proc.pid):
        if proc.poll() is not None:
            raise AssertionError("rc {} before blocking on the FIFO:\n{}".format(
                proc.returncode, output()))
        if time.monotonic() > deadline:
            raise AssertionError("not blocked on the FIFO after {} s:\n{}".format(
                timeout, output()))
        time.sleep(0.05)
