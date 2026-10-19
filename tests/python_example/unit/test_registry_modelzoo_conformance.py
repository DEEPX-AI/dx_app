# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""``config/model_registry.json`` must agree with the vendored dx-modelzoo tree.

The registry is the single source of truth for (task, family, variant). Family
assignment cannot be derived -- dx-modelzoo folds depth/width into the family
(resnet50 -> resnet, vgg16-bn -> vgg) -- so it is joined in from a committed
snapshot of the upstream tree and pinned here. Without this test the mapping
drifts silently the first time either side gains a model.

``variant`` is the identity used by ``--variant`` and by
``assets/models/<variant>.dxnn``; the four entries where it cannot equal the
dxnn stem are enumerated in EXCEPTIONS and nowhere else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_helpers.constants import EXCLUDED_EXAMPLE_VARIANTS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "config" / "model_registry.json"
SNAPSHOT = PROJECT_ROOT / "tests" / "data" / "modelzoo_cv_tree.json"

NEW_FIELDS = ("variant", "family", "task", "task_legacy", "image_only",
              "zoo_canonical", "alias_of")

# Added with the 2_5_0 additions: 143 of the 499 variants have no published .dxnn, so
# "is there a file for this" has to be a declared fact rather than an inference from
# whatever happens to be in assets/models on one machine.
PUBLICATION_FIELDS = ("published",)

# model_name -> (variant, task, family). The ONLY entries allowed to miss the
# dx-modelzoo snapshot. SCRFD500M_PPU and YOLOV5Pose_PPU are no longer rows;
# their example sources stay on disk and are listed in EXCLUDED_EXAMPLE_VARIANTS.
EXCEPTIONS: dict[str, tuple[str, str, str]] = {}
# efficientnet_lite0 was here while dx_app called the model 256x256 and the snapshot
# called it 224x224. The two .dxnn files are BYTE-IDENTICAL on the CDN (md5
# 5c966e8e...) and the binary self-reports [1, 224, 224, 3], so 256x256 was simply a
# wrong name. It now resolves from the snapshot like every other model, which is why
# it is neither an exception nor a SNAPSHOT_GAP any more.

# RETENTION POLICY: a conflicting example is KEPT, never deleted.
#
# When two entries land on the same variant, the non-canonical one becomes an
# *alias*: it keeps its own model_name as the legacy compat key and shares the
# canonical entry's variant/family/task/dxnn_file. It is NOT slated for removal
# at any phase -- the alignment renames and regroups examples, it does not
# delete them.
#
# There is no alias today. ``deit-b_384x384_distilled.dxnn`` had two rows --
# ``deit_base_distilled_2`` (owner) and ``deit_base384_distilled`` (alias) -- and the
# legacy ``_2`` name was dropped, leaving ``deit_base384_distilled`` as the sole owner.
# The non-distilled file stays with ``deitbase384`` only. The checks below still run:
# they keep the next alias honest instead of being deleted with this one.
#
# alias model_name -> canonical model_name it aliases.
ALIASES: dict[str, str] = {}

# In the snapshot but intentionally not in dx_app: no example, not on disk.
# Empty since the 2_5_0 revision: efficientnet-lite0 is 224x224 there, and dx_app was
# the only side still on 256x256 -- the gap WAS the disagreement.
SNAPSHOT_GAPS: set[str] = set()

# dx-modelzoo family renames. CMake target names are global, so two families sharing a
# name across tasks collide as ``<family>_sync``. Exactly one collision exists, and
# dx-modelzoo's own convention (yolo26-seg, yolov5-face, yolov5-pose) is to suffix the
# sibling by task -- it just did not apply that to casvit.
FAMILY_OVERRIDES: dict[tuple[str, str], str] = {
    ("semantic_segmentation", "casvit"): "casvit_seg",
}

# variant -> the task DX Model Zoo's own page assigns it, where that differs from the
# harvested dx-modelzoo GitHub tree. The snapshot is a RECORD of upstream (see its
# `source`/`ref`/`harvested_at`), so it is not edited to reflect our decisions; the
# disagreement is declared here instead, with its reason.
#
# The zoo page introduced four task categories in release 2_5_0 that the harvested tree
# predates. Both EigenPlaces variants were filed as `super_resolution` -- not a
# considered choice but the only slot the legacy task tables had for a bare embedding,
# and it left a place-recognition model defaulting to sample/img/face_pair. The zoo
# lists them under Visual Place Recognition, which is what they are.
#
# Only these two need an entry: the other five re-classified variants are 2_5_0
# additions, absent from the harvested tree, so the provisional table already carries
# their task and is updated in place by the emitter.
ZOO_PAGE_TASK: dict[str, str] = {
    "eigenplaces-resnet18_512x512": "visual_place_recognition",
    "eigenplaces-resnet50_512x512": "visual_place_recognition",
}


def _snake(family: str) -> str:
    """dx-modelzoo family -> dx_app family directory (import-safe)."""
    return family.replace("-", "_").replace(".", "_")


def _family_for(task: str, family: str) -> str:
    return FAMILY_OVERRIDES.get((task, family), _snake(family))


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snapshot() -> dict[str, dict]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))["variants"]


@pytest.fixture(scope="module")
def provisional() -> dict[str, dict]:
    """Our own (task, family) assignments for variants upstream has not filed.

    dx-modelzoo carries none of the 147 DX Model Zoo 2_5_0 additions (harvested
    2026-09-22: 350 variants, zero overlap), so for those the snapshot cannot be the
    oracle. The provisional table is, and unlike ``variants`` it already holds the
    dx_app-resolved family, so no FAMILY_OVERRIDES pass is applied to it.
    """
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))["provisional"]["variants"]


def test_every_entry_has_the_new_fields(registry):
    missing = {
        e["model_name"]: [f for f in NEW_FIELDS if f not in e]
        for e in registry
        if any(f not in e for f in NEW_FIELDS)
    }
    assert not missing, f"entries missing dx-modelzoo fields: {missing}"


def test_entry_key_set_is_exactly_the_expected_fields(registry):
    """Pin the full key set so neither a stale nor a stray field goes unnoticed."""
    expected = {
        "model_name", "dxnn_file", "original_name", "csv_task", "add_model_task",
        "postprocessor", "input_width", "input_height", "config", "source", "supported",
        *NEW_FIELDS, *PUBLICATION_FIELDS,
    }
    bad = {e["model_name"]: sorted(set(e) ^ expected) for e in registry if set(e) != expected}
    assert not bad, f"unexpected key set (symmetric difference shown): {bad}"


def test_variant_is_unique_among_canonical_entries(registry):
    """Every non-alias entry owns its variant outright.

    Entries listed in ALIASES deliberately share another entry's variant. They
    are retained examples, not pending deletions, so they are excluded here
    rather than being given a synthetic variant name that no .dxnn backs.
    """
    seen: dict[str, str] = {}
    dupes: list[tuple[str, str, str]] = []
    for e in registry:
        if e["alias_of"] is not None:
            continue
        v = e["variant"]
        if v in seen:
            dupes.append((v, seen[v], e["model_name"]))
        seen[v] = e["model_name"]
    assert not dupes, f"duplicate variant keys among canonical entries: {dupes}"


def test_alias_entries_are_declared_and_consistent(registry):
    """An alias must point at a real entry and carry that entry's identity."""
    by_name = {e["model_name"]: e for e in registry}
    declared = {e["model_name"]: e["alias_of"]
                for e in registry if e["alias_of"] is not None}
    assert declared == ALIASES, (
        f"alias_of markers are {declared}, ALIASES declares {ALIASES}"
    )
    for name, target_name in ALIASES.items():
        alias, target = by_name.get(name), by_name.get(target_name)
        assert alias is not None, f"ALIASES lists {name}, not in the registry"
        assert target is not None, f"{name} aliases {target_name}, not in the registry"
        assert target["alias_of"] is None, (
            f"{name} aliases {target_name}, which is itself an alias -- no alias chains"
        )
        assert alias["dxnn_file"] == target["dxnn_file"], (
            f"{name} claims to alias {target_name} but the .dxnn differs: "
            f"{alias['dxnn_file']} vs {target['dxnn_file']}"
        )
        for field in ("variant", "family", "task"):
            assert alias[field] == target[field], (
                f"{name}.{field}={alias[field]!r} must match {target_name}.{field}={target[field]!r}"
            )


def test_alias_entries_are_retained_not_deleted(registry):
    """Retention must be real, not just a label.

    The alignment regroups examples -- 353 per-variant directories became 89
    ``<task>/<family>/`` directories -- but it never deletes one. Post-restructure an
    alias is retained when its family directory exists in BOTH trees, its variant
    config is present, and the registry entry still resolves. Checking for the old
    per-variant directory would now fail for all 353 entries, not just aliases, so the
    test follows the variant rather than the legacy path.
    """
    by_name = {e["model_name"]: e for e in registry}
    for name, target_name in ALIASES.items():
        entry = by_name.get(name)
        assert entry is not None, f"alias {name} was dropped from the registry"
        for tree in ("python_example", "cpp_example"):
            fam = PROJECT_ROOT / "src" / tree / entry["task"] / entry["family"]
            assert fam.is_dir(), f"alias {name}: family dir missing -> {fam}"
            cfg = fam / entry["variant"] / "config.json"
            assert cfg.is_file(), f"alias {name}: variant config missing -> {cfg}"
        # The alias shares its canonical entry's variant, so the config it points at is
        # the canonical one -- that sharing is the whole point of an alias.
        assert entry["variant"] == by_name[target_name]["variant"]


def test_the_real_distilled_384_model_owns_its_dxnn_stem(registry):
    """deit-b_384x384_distilled.dxnn has one canonical owner.

    ``deit_base384_distilled`` is that owner and carries no alias: the legacy
    ``deit_base_distilled_2`` row was removed. It must own the distilled file,
    not deit-b_384x384.
    """
    owners = [e for e in registry if e["dxnn_file"] == "deit-b_384x384_distilled.dxnn"]
    canonical = [e for e in owners if e["alias_of"] is None]
    aliases = [e for e in owners if e["alias_of"] is not None]
    assert [e["model_name"] for e in canonical] == ["deit_base384_distilled"], (
        f"canonical owner must be deit_base384_distilled, got "
        f"{[e['model_name'] for e in canonical]}"
    )
    assert canonical[0]["variant"] == "deit-b_384x384_distilled"
    assert aliases == [], f"the distilled 384 model must have no alias row, got {aliases}"
    plain = [e for e in registry if e["dxnn_file"] == "deit-b_384x384.dxnn"]
    assert [e["model_name"] for e in plain] == ["deitbase384"]


def test_variant_matches_dxnn_stem_except_documented(registry):
    bad = [
        (e["model_name"], e["variant"], e["dxnn_file"])
        for e in registry
        if e["variant"] != e["dxnn_file"][: -len(".dxnn")]
        and e["model_name"] not in EXCEPTIONS
        and e["model_name"] not in ALIASES
    ]
    assert not bad, f"variant != dxnn stem outside EXCEPTIONS: {bad}"


def test_family_names_are_unique_across_tasks(registry):
    """CMake target names are global, so <family>_sync must be unambiguous.

    Two families sharing a name in different tasks would produce duplicate CMake
    targets. FAMILY_OVERRIDES exists to break exactly that; this test proves it is
    sufficient and stays sufficient as families are added.
    """
    import collections
    fams = {(e["task"], e["family"]) for e in registry}
    counts = collections.Counter(f for _, f in fams)
    clashes = {
        f: sorted(t for t, ff in fams if ff == f) for f, n in counts.items() if n > 1
    }
    assert not clashes, (
        f"family name(s) used in more than one task: {clashes} -- "
        "add an entry to FAMILY_OVERRIDES above"
    )


def test_task_and_family_match_the_snapshot(registry, snapshot, provisional):
    """Every entry's (task, family) must come from a declared table, never from nowhere.

    Upstream-published variants are checked against the harvested snapshot through
    FAMILY_OVERRIDES; the 2_5_0 additions are checked against the provisional table
    verbatim. An entry in neither table is the real failure this guards against.
    """
    bad = []
    for e in registry:
        if e["model_name"] in EXCEPTIONS:
            continue
        upstream = snapshot.get(e["variant"])
        if upstream is not None:
            task = ZOO_PAGE_TASK.get(e["variant"], upstream["task"])
            want = (task, _family_for(upstream["task"], upstream["family"]))
        elif e["variant"] in provisional:
            ours = provisional[e["variant"]]
            want = (ours["task"], ours["family"])
        else:
            bad.append((e["model_name"], e["variant"],
                        "in neither the snapshot nor the provisional table"))
            continue
        got = (e["task"], e["family"])
        if got != want:
            bad.append((e["model_name"], got, want))
    assert not bad, f"task/family disagree with their declared table: {bad}"


def _on_disk_variant_families(tree: str) -> dict[str, set[tuple[str, str]]]:
    """``variant stem -> {(task, family)}`` for every config in one example tree.

    The glob is ``<task>/<family>/<variant>/config.json``. The family directory is
    what the entry script loads; a second copy under another family is a second
    recipe, not a duplicate name.
    """
    locations: dict[str, set[tuple[str, str]]] = {}
    root = PROJECT_ROOT / "src" / tree
    for path in root.glob("*/*/*/config.json"):
        task_name = path.parent.parent.parent.name
        family_name = path.parent.parent.name
        locations.setdefault(path.parent.name, set()).add((task_name, family_name))
    return locations


def test_variant_config_lives_only_in_the_registry_model_folder(registry):
    """Each variant config exists only under the (task, family) the registry assigns.

    Entry scripts load ``<family>/<variant>/config.json``. Reassigning a stem in
    the registry and writing the new config does not remove the old file, and a
    stem-set check (zoo page, manifest, registry dxnn_file) still passes because
    both copies share a stem. The leftover is what
    ``yolo_preopt_sync.py --variant yolo11-l_640x640_pre-optimized`` used to load:
    a PreoptDetectionPostprocessor recipe for a dense YOLO head.
    """
    assigned: dict[str, set[tuple[str, str]]] = {}
    for entry in registry:
        assigned.setdefault(entry["variant"], set()).add((entry["task"], entry["family"]))

    misplaced: list[str] = []
    for tree in ("python_example", "cpp_example"):
        found = _on_disk_variant_families(tree)
        stems = set(assigned) | set(found)
        for stem in sorted(stems):
            if stem in EXCLUDED_EXAMPLE_VARIANTS:
                continue
            want = assigned.get(stem, set())
            got = found.get(stem, set())
            if got == want:
                continue
            misplaced.append(f"{tree}/{stem}: on disk {sorted(got)}, registry {sorted(want)}")
    assert not misplaced, (
        "variant config outside the registry-assigned family:\n  "
        + "\n  ".join(misplaced)
    )


def test_exceptions_carry_their_declared_values(registry):
    by_name = {e["model_name"]: e for e in registry}
    for name, (variant, task, family) in EXCEPTIONS.items():
        e = by_name.get(name)
        assert e is not None, f"EXCEPTIONS lists {name}, which is not in the registry"
        assert (e["variant"], e["task"], e["family"]) == (variant, task, family), (
            f"{name}: got {(e['variant'], e['task'], e['family'])}, want {(variant, task, family)}"
        )
        assert e["zoo_canonical"] is False, f"{name} must be zoo_canonical=False"


def test_zoo_canonical_flag_agrees_with_snapshot_membership(registry, snapshot):
    bad = [
        (e["model_name"], e["zoo_canonical"], e["variant"] in snapshot)
        for e in registry
        if e["zoo_canonical"] is not (e["variant"] in snapshot)
    ]
    assert not bad, f"zoo_canonical disagrees with snapshot membership: {bad}"


def test_snapshot_coverage(registry, snapshot):
    covered = {e["variant"] for e in registry}
    uncovered = set(snapshot) - covered - SNAPSHOT_GAPS
    assert not uncovered, f"dx-modelzoo variants with no registry entry: {sorted(uncovered)}"


def test_task_legacy_preserves_the_old_task(registry):
    bad = [
        (e["model_name"], e["task_legacy"], e["add_model_task"])
        for e in registry
        if e["task_legacy"] != e["add_model_task"]
    ]
    assert not bad, f"task_legacy must mirror add_model_task: {bad}"
