# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The demo registry exists TWICE, and the two copies must stay identical.

`run_demo.sh` (Linux) carries ten parallel bash arrays; `scripts/run_demo.py` (which
`run_demo.bat` delegates to, and which provides `--all`) carries the same data as a
list of tuples. Nothing linked them, and they had drifted on six fields -- three of
them behavioural, so a Linux user and a Windows user ran different demos:

    [6]  face alignment input : sample_face_a1.jpg  vs  face_pair/1_reference.jpg
                                (the same file, md5 441431812074..., so cosmetic)
    [18] SuperPoint video     : dance-solo.mov      vs  blackbox-city-road2.mov
    [22] hand detection input : sample_hand.jpg     vs  sample_person_a2.jpg

The last one is the clearest: a hand DETECTOR was being demoed on a full-body photo.

These tests pin the two copies together and pin every reference to something that
actually exists, so the next edit to one file fails here instead of in front of a user.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_DEMO_SH = PROJECT_ROOT / "run_demo.sh"
RUN_DEMO_PY = PROJECT_ROOT / "scripts" / "run_demo.py"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

FIELDS = ("label", "group", "cpp_base", "py_dir", "py_base", "model", "video",
          "image", "py_async", "image_only")


@pytest.fixture(scope="module")
def py_demos() -> list[tuple]:
    from run_demo import DEMOS
    return DEMOS


@pytest.fixture(scope="module")
def sh_demos() -> list[list[str]]:
    """The bash arrays, read by sourcing them -- not by regex.

    Sourcing is what `run_demo.sh` itself does with these values, so a quoting or
    line-continuation mistake shows up here the same way it would at runtime.
    """
    if not RUN_DEMO_SH.is_file():
        pytest.skip("run_demo.sh is not in this checkout")
    text = RUN_DEMO_SH.read_text(encoding="utf-8").splitlines()
    end = next(i for i, l in enumerate(text) if l.startswith("DEMO_COUNT="))
    body = "\n".join(text[:end])
    script = body + """
n=${#DEMO_LABELS[@]}
for ((i=0;i<n;i++)); do
  printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \
    "${DEMO_LABELS[$i]}" "${DEMO_GROUPS[$i]}" "${DEMO_CPP_BASE[$i]}" \
    "${DEMO_PY_DIR[$i]}" "${DEMO_PY_BASE[$i]}" "${DEMO_MODEL[$i]}" \
    "${DEMO_VIDEO[$i]}" "${DEMO_IMAGE[$i]}" "${DEMO_PY_ASYNC[$i]}" \
    "${DEMO_IMAGE_ONLY[$i]}"
done
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         cwd=PROJECT_ROOT)
    assert out.returncode == 0, out.stderr
    return [l.split("\t") for l in out.stdout.splitlines()]


def _norm(field: str, value: str) -> str:
    # labels are column-aligned with runs of spaces in both files
    return re.sub(r"\s+", " ", value).strip() if field == "label" else value


def _as_row(demo: tuple) -> list[str]:
    return [demo[0], demo[1], demo[2], demo[3], demo[4], demo[5], demo[6] or "",
            demo[7], "full" if demo[8] else "no_py_async", "1" if demo[9] else "0"]


def test_both_registries_hold_the_same_number_of_demos(sh_demos, py_demos):
    assert len(sh_demos) == len(py_demos)


def test_every_field_of_every_demo_agrees(sh_demos, py_demos):
    assert len(sh_demos) == len(py_demos), "compare lengths first"
    bad = []
    for i, (s, p) in enumerate(zip(sh_demos, py_demos)):
        for field, a, b in zip(FIELDS, s, _as_row(p)):
            if _norm(field, a) != _norm(field, b):
                bad.append(f"[{i}] {field}: run_demo.sh={a!r} run_demo.py={b!r}")
    assert not bad, "the two demo registries disagree:\n  " + "\n  ".join(bad)


def test_the_bash_arrays_are_all_the_same_length(sh_demos):
    """Ten parallel arrays: one short array silently shifts every later field."""
    assert all(len(row) == len(FIELDS) for row in sh_demos)


def test_groups_are_contiguous(py_demos):
    """Stage 1 prints a header whenever the group CHANGES, so a group split across
    non-adjacent entries prints its header twice."""
    seen, previous, repeated = set(), None, []
    for demo in py_demos:
        group = demo[1]
        if group != previous:
            if group in seen:
                repeated.append(group)
            seen.add(group)
            previous = group
    assert not repeated, f"non-contiguous groups print a duplicate header: {repeated}"


def test_every_demo_input_image_exists(py_demos):
    missing = [d[7] for d in py_demos if not (PROJECT_ROOT / d[7]).exists()]
    assert not missing, f"demo input images that do not exist: {missing}"


def test_every_video_capable_demo_names_a_video(py_demos):
    """Only this direction matters. A video-capable demo with no video breaks Stage 3,
    which reads DEMO_VIDEO[task] the moment the user picks "video".

    The reverse is NOT asserted: five image-only demos (14, 15, 16, 19, 21) still carry
    a leftover video path from before they were marked image-only. It is dead data --
    the value is never read for an image-only task -- so requiring it to be empty would
    be inventing a convention rather than pinning one.
    """
    wrong = [f"[{i}] {d[0].strip()}" for i, d in enumerate(py_demos)
             if not d[9] and not d[6]]
    assert not wrong, f"video-capable demos naming no video: {wrong}"


def test_every_python_variant_the_menu_offers_actually_exists(py_demos):
    """The mode menu offers async only when py_async is set; every offered mode must
    resolve to a file, or the user picks a mode and gets a traceback."""
    root = PROJECT_ROOT / "src" / "python_example"
    missing = []
    for i, d in enumerate(py_demos):
        stem = Path(d[5]).stem
        kinds = ["sync", "sync_cpp_postprocess"]
        if d[8]:
            kinds += ["async", "async_cpp_postprocess"]
        for kind in kinds:
            script = root / d[3] / stem / f"{stem}_{kind}.py"
            if not script.is_file():
                missing.append(f"[{i}] {script.relative_to(PROJECT_ROOT)}")
    assert not missing, "demo modes with no script:\n  " + "\n  ".join(missing)


def test_every_demo_model_is_declared_and_downloadable(py_demos):
    """A demo whose model is absent from the manifest cannot be auto-downloaded, and
    run_demo's first action is to fetch the missing ones."""
    import json
    manifest = {Path(r["dxnn_url"]).name
                for r in json.loads((PROJECT_ROOT / "scripts"
                                     / "modelzoo_manifest.json").read_text())
                if r.get("dxnn_url")}
    registry = {r["dxnn_file"] for r in json.loads(
        (PROJECT_ROOT / "config" / "model_registry.json").read_text())}
    bad = [d[5] for d in py_demos if d[5] not in manifest or d[5] not in registry]
    assert not bad, f"demo models missing from the manifest/registry: {bad}"
