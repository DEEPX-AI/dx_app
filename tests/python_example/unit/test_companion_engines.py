# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Companion engines: one factory, several .dxnn files, one `-m` on the command line.

EfficientAD ships as three networks that must all see the same frame. Adding a
``--model2/--model3`` flag would have changed the CLI contract every caller relies on
-- ``run_demo.sh`` passes exactly one ``-m``, and so does every sweep and test -- so
the FACTORY declares what else it needs and the runner resolves those files next to
the primary one.

The failure this design has to avoid is silence. A companion that cannot be found
must stop the run naming the file: an ensemble that quietly degrades to one network
produces a plausible-looking heatmap that means something entirely different, which
is precisely the bug that motivated this work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.runner.sync_runner import SyncRunner, resolve_companion_models  # noqa: E402


class _PlainFactory:
    """A factory that needs nothing else -- i.e. 110 of the 111 families."""


class _TripleFactory:
    def get_companion_models(self, primary_path):
        stem = Path(primary_path).name
        return [("teacher", stem.replace("-student", "-teacher")),
                ("autoencoder", stem.replace("-student", "-autoencoder"))]


class _FakeEngine:
    def __init__(self, tag: str):
        self.tag = tag
        self.seen = []

    def run(self, inputs):
        self.seen.append(inputs[0])
        return [np.full((1, 2), ord(self.tag[0]), dtype=np.float32)]


def _make_triple(tmp_path: Path) -> Path:
    for role in ("student", "teacher", "autoencoder"):
        (tmp_path / f"efficientad-m-{role}_256x256.dxnn").write_bytes(b"DXNN")
    return tmp_path / "efficientad-m-student_256x256.dxnn"


def test_a_factory_that_declares_nothing_gets_no_companions(tmp_path):
    assert resolve_companion_models(_PlainFactory(), str(tmp_path / "x.dxnn")) == []


def test_companions_are_resolved_next_to_the_primary_model(tmp_path):
    primary = _make_triple(tmp_path)
    resolved = resolve_companion_models(_TripleFactory(), str(primary))
    assert [role for role, _ in resolved] == ["teacher", "autoencoder"]
    assert [Path(p).name for _, p in resolved] == [
        "efficientad-m-teacher_256x256.dxnn",
        "efficientad-m-autoencoder_256x256.dxnn",
    ]
    assert all(Path(p).is_file() for _, p in resolved)


def test_a_missing_companion_names_the_file_and_the_role(tmp_path):
    primary = tmp_path / "efficientad-m-student_256x256.dxnn"
    primary.write_bytes(b"DXNN")
    (tmp_path / "efficientad-m-teacher_256x256.dxnn").write_bytes(b"DXNN")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_companion_models(_TripleFactory(), str(primary))
    message = str(exc.value)
    assert "efficientad-m-autoencoder_256x256.dxnn" in message
    assert "autoencoder" in message


def test_infer_returns_the_primary_outputs_first_then_each_companion():
    """The order IS the contract: teacher and autoencoder are both (1,384,H,W)."""
    runner = SyncRunner(_TripleFactory())
    runner.ie = _FakeEngine("student")
    runner._companion_engines = [("teacher", _FakeEngine("teacher")),
                                 ("autoencoder", _FakeEngine("autoencoder"))]
    runner._input_dtype = np.float32
    frame = np.zeros((4, 4, 3), dtype=np.float32)

    outputs = runner.infer(frame)

    assert [float(o[0, 0]) for o in outputs] == [ord("s"), ord("t"), ord("a")]


def test_every_engine_sees_the_same_prepared_tensor():
    """Preprocessing once and sharing the tensor is the point of an ensemble."""
    runner = SyncRunner(_TripleFactory())
    runner.ie = _FakeEngine("student")
    companions = [("teacher", _FakeEngine("teacher"))]
    runner._companion_engines = companions
    runner._input_dtype = np.float32
    frame = np.arange(48, dtype=np.float32).reshape(4, 4, 3)

    runner.infer(frame)

    assert np.array_equal(runner.ie.seen[0], companions[0][1].seen[0])
