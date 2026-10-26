# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""``image_only`` in the registry must reproduce the task-keyed runner behavior.

``sync_runner._IMAGE_ONLY_TASKS`` keys image-only-ness on the task name. The
dx-modelzoo task alignment (Phase 1) deletes three of those task names and moves
two of their models into video-capable tasks:

    reid/casvit_*           -> image_classification  (110 other models need video)
    embedding/eigenplaces-* -> super_resolution      (6 other models need video)

So the flag has to move to the variant before the tasks are renamed. This test
pins the migrated per-variant flag to what the runner does *today*, making the
rename provably behavior-preserving.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "config" / "model_registry.json"
SYNC_RUNNER = (PROJECT_ROOT / "src" / "python_example" / "common" / "runner" / "sync_runner.py")

# Plain mirror of sync_runner._IMAGE_ONLY_TASKS -- both sides spell every task the
# same way, so no translation is applied.
IMAGE_ONLY_LEGACY_TASKS = frozenset({
    # legacy task_legacy spellings -- these are the ones registry rows actually carry
    "embedding", "reid", "attribute_recognition",
    "object_pose_estimation", "3d_object_detection",
    # current task names the runner also treats as image-only. No registry row has
    # one of these as task_legacy, so they do not affect
    # test_image_only_matches_legacy_task_membership; they are here because the
    # runner's set is consulted with either spelling depending on the variant.
    "face_recognition", "person_attribute", "face_attribute",
    "person_reid", "image_retrieval", "visual_place_recognition",
})


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_image_only_matches_legacy_task_membership(registry):
    bad = [
        (e["model_name"], e["task_legacy"], e["image_only"])
        for e in registry
        if e["image_only"] != (e["task_legacy"] in IMAGE_ONLY_LEGACY_TASKS)
    ]
    assert not bad, f"image_only diverges from the legacy task set: {bad}"


def test_image_only_count_is_pinned(registry):
    """17 at the Phase 0 migration, 22 after the 2_5_0 additions.

    The five new ones all land in a legacy task the runner already treats as
    image-only, so the count moves without the RULE moving -- which is exactly what
    test_image_only_matches_legacy_task_membership above checks:

        clip-img_resnet50_224x224_openai                    legacy embedding
        clip-img_vit-b16-quickgelu_224x224_metaclip-fullcc  legacy embedding
        clip-text_resnet50_77x512_openai                    legacy embedding
        pp-shituv2-feature-extraction_224x224               legacy embedding
        repvgg-a0-reid_256x128                              legacy reid

    The pin stays because it is the tripwire for a flag being set by accident: a
    variant silently becoming image-only loses video input with no error.
    """
    flagged = [e["variant"] for e in registry if e["image_only"]]
    assert len(flagged) == 22, (
        f"expected 22 image-only variants, got {len(flagged)}: {sorted(flagged)}"
    )


def test_runner_image_only_set_still_matches_this_mirror():
    """Fail loudly if the runner's set changes without this mirror being updated."""
    text = SYNC_RUNNER.read_text(encoding="utf-8")
    start = text.index("_IMAGE_ONLY_TASKS = {")
    literal = text[start : text.index("}", start) + 1]
    names = {tok.strip().strip('"').strip("'")
             for tok in literal.split("{", 1)[1].rstrip("}").split(",")}
    names = {n for n in names if n}
    expected = set(IMAGE_ONLY_LEGACY_TASKS)
    assert names == expected, (
        f"sync_runner._IMAGE_ONLY_TASKS is {sorted(names)}, this test mirrors "
        f"{sorted(expected)} -- update IMAGE_ONLY_LEGACY_TASKS and the migration "
        "script together"
    )


def test_relocated_comparison_models_keep_image_only(registry):
    """A variant that compares against a stored set must stay image-only after a move.

    These four were relocated by task re-classifications, and the flag is the only
    thing stopping a video source being accepted for a model that has nothing to
    compare frame-by-frame.

    The two EigenPlaces variants were listed here as `super_resolution`, which was
    never a considered assignment -- it was the only slot the legacy task tables had
    for a bare embedding, and it left a place-recognition model defaulting to
    sample/img/face_pair. DX Model Zoo 2_5_0 gave them their own category and they now
    sit in `visual_place_recognition`, which is itself image-only; casvit's
    `image_classification` is video-capable, so for those two the flag is still the
    only guard. Both cases are pinned, because the hazard is a flag quietly flipping.
    """
    by_variant = {e["variant"]: e for e in registry}
    movers = [
        ("casvit-t_224x224", "image_classification"),
        ("casvit-m_224x224", "image_classification"),
        ("eigenplaces-resnet18_512x512", "visual_place_recognition"),
        ("eigenplaces-resnet50_512x512", "visual_place_recognition"),
    ]
    for variant, new_task in movers:
        e = by_variant.get(variant)
        assert e is not None, f"{variant} missing from the registry"
        assert e["task"] == new_task, f"{variant}: task is {e['task']}, expected {new_task}"
        assert e["image_only"] is True, (
            f"{variant} sits in task {new_task} and MUST keep image_only=True"
        )
