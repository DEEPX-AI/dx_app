"""OpenCV HighGUI warnings that video mode prints before the first window.

The opencv wheel points ``QT_QPA_FONTDIR`` at ``cv2/qt/fonts``, which is not
shipped, and Qt 5.15 warns when a GNOME session is Wayland even though the
wheel only has the xcb plugin. ``show_output`` corrects both before
``namedWindow``.
"""

import os
import sys

import numpy as np
import pytest

import common.utility.common_util as util


def _require_system_font_dir():
    if any(os.path.isdir(path) for path in util._FONT_DIR_CANDIDATES):
        return
    pytest.skip("no system font directory on this machine")


def test_missing_font_dir_is_replaced(monkeypatch):
    _require_system_font_dir()
    monkeypatch.setenv("QT_QPA_FONTDIR", "/no/such/cv2/qt/fonts")
    util.prepare_gui_backend()
    font_dir = os.environ["QT_QPA_FONTDIR"]
    assert font_dir != "/no/such/cv2/qt/fonts"
    assert os.path.isdir(font_dir)


def test_existing_font_dir_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_FONTDIR", str(tmp_path))
    util.prepare_gui_backend()
    assert os.environ["QT_QPA_FONTDIR"] == str(tmp_path)


def test_non_linux_leaves_font_dir_alone(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("QT_QPA_FONTDIR", "/no/such/cv2/qt/fonts")
    util.prepare_gui_backend()
    assert os.environ["QT_QPA_FONTDIR"] == "/no/such/cv2/qt/fonts"


def _show_once(monkeypatch, seen):
    monkeypatch.setattr(util, "_window_initialized", False)
    monkeypatch.setattr(util, "_get_screen_resolution", lambda: (640, 480))

    def fake_named(_name, _flags):
        seen["session"] = os.environ.get("XDG_SESSION_TYPE")
        seen["font"] = os.environ.get("QT_QPA_FONTDIR")

    monkeypatch.setattr(util.cv2, "namedWindow", fake_named)
    monkeypatch.setattr(util.cv2, "resizeWindow", lambda *_a, **_k: None)
    monkeypatch.setattr(util.cv2, "imshow", lambda *_a, **_k: None)
    util.show_output(np.zeros((8, 8, 3), np.uint8))


def test_gnome_wayland_is_xcb_only_while_the_window_opens(monkeypatch):
    _require_system_font_dir()
    monkeypatch.setenv("QT_QPA_FONTDIR", "/no/such/cv2/qt/fonts")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")
    monkeypatch.delenv("DESKTOP_SESSION", raising=False)
    seen = {}
    _show_once(monkeypatch, seen)
    assert seen["session"] == "x11"
    assert os.path.isdir(seen["font"])
    assert os.environ["XDG_SESSION_TYPE"] == "wayland"


def test_other_desktops_keep_their_session_type(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("DESKTOP_SESSION", "plasma")
    seen = {}
    _show_once(monkeypatch, seen)
    assert seen["session"] == "wayland"
    assert os.environ["XDG_SESSION_TYPE"] == "wayland"
