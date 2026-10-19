# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The DX Model Zoo's task column is the authority for which task an example lives under.

DX Model Zoo 2_5_0 introduced four categories that had no directory here, and every
model in them was filed under whatever the legacy task tables could accommodate. The
worst of it was not cosmetic: both EigenPlaces variants and PP-ShiTuV2 sat in
``super_resolution``, with ``default_image`` pointing at ``sample/img/face_pair`` -- a
place-recognition model aimed at a pair of face photographs.

    Image Retrieval           clip-text_resnet50_77x512_openai
    Image Matting             ppmatting-hrnet-w48-{composition,distinctions}_512x512
    Visual Place Recognition  eigenplaces-resnet{18,50}_512x512
                              pp-shituv2-feature-extraction_224x224
    Person ReID               repvgg-a0-reid_256x128

These tests pin that assignment in the registry and in both example trees, so the next
re-classification cannot quietly regress it.

ONE DELIBERATE DEPARTURE, asserted rather than tolerated:
``clip-img_resnet50_224x224_openai``. The page lists it under Zero Shot Image
Classification, and it lives under ``image_retrieval`` instead, because retrieval needs
both halves of one embedding space and this is the text tower's own OpenAI RN50 pair.
It also had no working example anywhere before the move -- its config applied open_clip
mean/std to a .dxnn whose input is uint8, which the engine rejects outright, and the
prompt bank beside it is 512-d ViT-B/32 against this model's 1024-d output.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT_ROOT / "config" / "model_registry.json"
PY_TREE = PROJECT_ROOT / "src" / "python_example"
CPP_TREE = PROJECT_ROOT / "src" / "cpp_example"

# Zoo category -> the task directory it maps to here.
CATEGORY_TASK = {
    "Image Retrieval": "image_retrieval",
    "Image Matting": "image_matting",
    "Visual Place Recognition": "visual_place_recognition",
    "Person ReID": "person_reid",
}

# Listed by the page under another category, deliberately placed here. See the
# module docstring.
EXCEPTIONS = {"clip-img_resnet50_224x224_openai": "image_retrieval"}

# What the zoo lists under each new category (the table in the module docstring).
CATEGORY_MEMBERS = {
    "Image Retrieval": ["clip-text_resnet50_77x512_openai"],
    "Image Matting": ["ppmatting-hrnet-w48-composition_512x512",
                      "ppmatting-hrnet-w48-distinctions_512x512"],
    "Visual Place Recognition": ["eigenplaces-resnet18_512x512",
                                 "eigenplaces-resnet50_512x512",
                                 "pp-shituv2-feature-extraction_224x224"],
    "Person ReID": ["repvgg-a0-reid_256x128"],
}


@pytest.fixture(scope="module")
def registry_task() -> dict:
    rows = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {r["variant"]: r["task"] for r in rows}


@pytest.mark.parametrize("category", sorted(CATEGORY_TASK))
def test_registry_task_matches_the_zoo_category(category, registry_task):
    expected = CATEGORY_TASK[category]
    wrong = {s: registry_task.get(s) for s in CATEGORY_MEMBERS[category]
             if registry_task.get(s) != expected}
    assert not wrong, (
        f"{category!r} should map to task {expected!r}; these disagree: {wrong}")


@pytest.mark.parametrize("task", sorted(set(CATEGORY_TASK.values())))
def test_both_trees_carry_a_directory_for_the_task(task):
    for tree in (PY_TREE, CPP_TREE):
        d = tree / task
        assert d.is_dir(), f"missing {d.relative_to(PROJECT_ROOT)}"
        families = [p for p in d.iterdir() if p.is_dir()]
        assert families, f"{d.relative_to(PROJECT_ROOT)} has no family directory"


@pytest.mark.parametrize("variant,task", sorted(EXCEPTIONS.items()))
def test_documented_exceptions_are_where_this_module_says(variant, task,
                                                          registry_task):
    assert registry_task.get(variant) == task


def test_every_new_task_variant_has_an_example_in_both_trees(registry_task):
    """A task directory is only real if each of its variants can actually be run."""
    wanted = {v: t for v, t in registry_task.items()
              if t in set(CATEGORY_TASK.values())}
    assert wanted, "no variants carry the new tasks"
    for variant, task in sorted(wanted.items()):
        for tree in (PY_TREE, CPP_TREE):
            hits = sorted(tree.glob(f"{task}/*/{variant}/config.json"))
            assert len(hits) == 1, (
                f"{variant}: expected exactly one "
                f"{tree.name}/{task}/*/{variant}/config.json, found {len(hits)}")
            cfg = json.loads(hits[0].read_text(encoding="utf-8"))
            assert cfg.get("task") == task, (
                f"{hits[0].relative_to(PROJECT_ROOT)} declares task "
                f"{cfg.get('task')!r}, registry says {task!r}")


def test_no_new_task_variant_still_points_at_the_face_pair_samples(registry_task):
    """The bug that started this: a VPR model defaulting to sample/img/face_pair.

    Retrieval, place recognition and ReID each need a query that their own gallery can
    answer. A face-pair folder is not one, and the example produced no output image at
    all because the comparison visualizer was waiting for a reference frame.
    """
    offenders = {}
    for variant, task in registry_task.items():
        if task not in set(CATEGORY_TASK.values()):
            continue
        for cfg in PY_TREE.glob(f"{task}/*/{variant}/config.json"):
            image = (json.loads(cfg.read_text(encoding="utf-8"))
                     .get("default_image") or "")
            if "face_pair" in image or "person_pair" in image:
                offenders[variant] = image
    assert not offenders, f"stale pair-comparison defaults: {offenders}"


def test_the_text_tower_promises_no_image_input():
    """``clip-text_resnet50_77x512_openai`` takes tokens, so it offers no default image.

    MEASURED: its input is float32 [1,77,512] token embeddings. Feeding it what an
    image preprocessor produces fails with "Input dtype mismatch for 'x': expected
    float32, got uint8", so ANY default image is a promise the example cannot keep --
    and the 49408x512 token-embedding table needed to encode text on device is far too
    large to ship. It stays as the embedding-contract example.
    """
    for tree in (PY_TREE, CPP_TREE):
        cfg = (tree / "image_retrieval" / "clip_rn50"
               / "clip-text_resnet50_77x512_openai" / "config.json")
        assert cfg.is_file(), cfg
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data.get("default_image") is None, (
            f"{cfg.relative_to(PROJECT_ROOT)} offers default_image "
            f"{data.get('default_image')!r}, which cannot be fed to a token-input model")


def test_the_gallery_path_agrees_between_its_two_readers():
    """One gallery file, named twice in each config, and the two names must match.

    The C++ ModelConfig is a FLAT parser -- it skips nested objects outright -- so it
    cannot see anything inside the `config` block the Python factory reads. The path is
    therefore written at the top level as well. Two copies is two chances to disagree,
    and a disagreement is invisible: each tree loads a gallery, ranks against it, and
    reports confident scores. Hence this test.
    """
    seen = 0
    for tree in (PY_TREE, CPP_TREE):
        for cfg in sorted(tree.glob("*/*/*/config.json")):
            data = json.loads(cfg.read_text(encoding="utf-8"))
            nested = (data.get("config") or {}).get("gallery")
            if nested is None and "gallery" not in data:
                continue
            seen += 1
            where = cfg.relative_to(PROJECT_ROOT)
            assert data.get("gallery") == nested, (
                f"{where}: top-level gallery {data.get('gallery')!r} != "
                f"config.gallery {nested!r}")
            assert str(nested).endswith(".bin"), (
                f"{where}: gallery {nested!r} is not the shared .bin format that both "
                f"trees read")
            assert (PROJECT_ROOT / str(nested)).is_file(), (
                f"{where}: gallery {nested} does not exist -- build it with "
                f"scripts/build_gallery_database.py")
    assert seen == 10, f"expected 5 variants x 2 trees to carry a gallery, saw {seen}"


def test_every_gallery_was_built_by_the_model_that_uses_it():
    """A gallery from another encoder ranks plausibly and wrongly, so pin the pairing."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))
    from common.processors.gallery_format import read_gallery

    for cfg in sorted(PY_TREE.glob("*/*/*/config.json")):
        data = json.loads(cfg.read_text(encoding="utf-8"))
        gallery = (data.get("config") or {}).get("gallery")
        if not gallery:
            continue
        variant = cfg.parent.name
        loaded = read_gallery(PROJECT_ROOT / gallery)
        assert loaded["model"] == variant, (
            f"{gallery} was built by {loaded['model']!r} but is configured for "
            f"{variant!r}")
        assert loaded["dim"] == loaded["embeddings"].shape[1]
        assert len(loaded["paths"]) == loaded["embeddings"].shape[0]
        assert len(loaded["labels"]) == loaded["embeddings"].shape[0]
