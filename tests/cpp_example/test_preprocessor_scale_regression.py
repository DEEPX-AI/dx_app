"""Golden checks for GrayscaleResizePreprocessor's per-axis scale.

DnCNN and ESPCN must keep the output captured after the per-axis scale fix.
SuperPoint keypoints on a non-4:3 frame must still land on corners.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import PROJECT_ROOT, resolve_bin_dir  # noqa: E402

cv2 = pytest.importorskip("cv2", reason="opencv-python required for output checks")

# ---------------------------------------------------------------------------
# golden layer -- runs the binaries
# ---------------------------------------------------------------------------

_DENOISE_IMG = "sample/img/sample_denoising.jpg"
_LOWRES_IMG = "sample/img/sample_lowres275x150.png"

# Captured on DX-M1 with the per-axis fix in place, and confirmed byte-identical
# to a build with the pre-fix header. shape is exact; mean/std carry a tolerance
# so a different JPEG encoder does not trip the test.
GOLDEN = {
    "dncnn_15":         dict(dxnn="dncnn-15_512x512.dxnn",   image=_DENOISE_IMG, shape=(512, 1024), mean=124.34, std=63.90),
    "dncnn_25":         dict(dxnn="dncnn-25_512x512.dxnn",   image=_DENOISE_IMG, shape=(512, 1024), mean=124.42, std=61.38),
    "dncnn_50":         dict(dxnn="dncnn-50_512x512.dxnn",   image=_DENOISE_IMG, shape=(512, 1024), mean=124.30, std=56.20),
    "dncnn_gray_blind": dict(dxnn="dncnn-gray_512x512.dxnn", image=_DENOISE_IMG, shape=(512, 1024), mean=124.65, std=56.97),
    "espcn_x3":         dict(dxnn="espcn-x3_17x17.dxnn",     image=_LOWRES_IMG,  shape=(450, 1654), mean=102.61, std=46.13),
    "espcn_x4":         dict(dxnn="espcn-x4_17x17.dxnn",     image=_LOWRES_IMG,  shape=(600, 2204), mean=102.82, std=45.79),
}

MEAN_TOL = 1.0   # grey levels
STD_TOL = 1.0


def _run_and_load(model: str, spec: dict, tmp_path: Path):
    """Run the model's <dxnn stem>_sync with --save and return the decoded output image."""
    exe = resolve_bin_dir() / f"{Path(spec['dxnn']).stem}_sync"
    if not exe.exists():
        pytest.skip(f"{exe.name} not built")
    dxnn = PROJECT_ROOT / "assets" / "models" / spec["dxnn"]
    if not dxnn.exists():
        pytest.skip(f"{spec['dxnn']} not downloaded")
    image = PROJECT_ROOT / spec["image"]
    if not image.exists():
        pytest.skip(f"{spec['image']} missing")

    result = subprocess.run(
        [str(exe), "-m", str(dxnn), "-i", str(image),
         "--no-display", "-s", "--save-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"{exe.name} exited {result.returncode}\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}"
    )

    saved = sorted(p for p in tmp_path.rglob("*") if p.suffix.lower() in (".jpg", ".png"))
    assert saved, f"{exe.name} --save produced no image under {tmp_path}"
    img = cv2.imread(str(saved[0]), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"could not decode {saved[0]}"
    return img


@pytest.mark.golden
@pytest.mark.parametrize("model", sorted(GOLDEN))
def test_restoration_output_unchanged_by_per_axis_scale(model, tmp_path):
    """DnCNN/ESPCN must stay bit-for-bit indifferent to the scale bookkeeping."""
    spec = GOLDEN[model]
    img = _run_and_load(model, spec, tmp_path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    assert grey.shape == spec["shape"], (
        f"{model} output geometry changed: {grey.shape} != {spec['shape']}. "
        "A scale-mapping regression shows up here first."
    )
    assert abs(float(grey.mean()) - spec["mean"]) <= MEAN_TOL, (
        f"{model} output content drifted: mean {grey.mean():.2f} vs "
        f"reference {spec['mean']:.2f} (tolerance {MEAN_TOL})"
    )
    assert abs(float(grey.std()) - spec["std"]) <= STD_TOL, (
        f"{model} output content drifted: std {grey.std():.2f} vs "
        f"reference {spec['std']:.2f} (tolerance {STD_TOL})"
    )


# The input MUST NOT share the model's 4:3 aspect ratio. On a 4:3 source the two
# branches of scaleKeypoint compute the same thing (scale_x == scale_y), so the
# bug is invisible: measured on the pre-fix binary, 768x576 sample_street scores
# 11.9x and passes, while 1920x1080 sample_crowd scores 1.3x and fails.
SUPERPOINT_IMAGE = "sample/img/sample_crowd.jpg"
MODEL_ASPECT = 640 / 480


@pytest.mark.contract
def test_superpoint_probe_image_is_not_model_aspect():
    """Guard the guard: a 4:3 probe image would make the test below vacuous."""
    image = PROJECT_ROOT / SUPERPOINT_IMAGE
    if not image.exists():
        pytest.skip(f"{SUPERPOINT_IMAGE} missing")
    img = cv2.imread(str(image))
    assert img is not None, f"could not decode {image}"
    h, w = img.shape[:2]
    assert abs((w / h) - MODEL_ASPECT) > 0.05, (
        f"{SUPERPOINT_IMAGE} is {w}x{h}, aspect {w / h:.3f}, which matches the "
        f"model's {MODEL_ASPECT:.3f}. On such an input scale_x == scale_y and both "
        "branches of scaleKeypoint agree, so test_superpoint_keypoints_land_on_corners "
        "would pass even with the per-axis factors removed. Pick a non-4:3 image."
    )


@pytest.mark.golden
def test_superpoint_keypoints_land_on_corners(tmp_path):
    """The positive half: the fix must keep SuperPoint keypoints on real corners.

    Measured on a 1920x1080 input: with the uniform fallback the rendered
    keypoints score 1.3x a random-position control on Shi-Tomasi corner response
    -- i.e. no better than chance. With the per-axis factors they score >20x.
    """
    exe = resolve_bin_dir() / "superpoint_480x640_sync"
    if not exe.exists():
        pytest.skip("superpoint_480x640_sync not built")
    dxnn = PROJECT_ROOT / "assets" / "models" / "superpoint_480x640.dxnn"
    if not dxnn.exists():
        pytest.skip("superpoint_480x640.dxnn not downloaded")
    image = PROJECT_ROOT / SUPERPOINT_IMAGE
    if not image.exists():
        pytest.skip(f"{SUPERPOINT_IMAGE} missing")

    result = subprocess.run(
        [str(exe), "-m", str(dxnn), "-i", str(image),
         "--no-display", "-s", "--save-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"{exe.name} exited {result.returncode}\n{result.stderr[-1500:]}"
    saved = sorted(p for p in tmp_path.rglob("*") if p.suffix.lower() in (".jpg", ".png"))
    assert saved, f"{exe.name} --save produced no image"

    out = cv2.imread(str(saved[0]))
    src = cv2.imread(str(image))
    src = cv2.resize(src, (out.shape[1], out.shape[0]))

    response = cv2.cornerMinEigenVal(
        cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32), 7, 3)
    response /= max(float(response.max()), 1e-12)

    # green dots are the keypoints the visualizer drew
    b, g, r = (out[:, :, i].astype(int) for i in range(3))
    mask = ((g > 170) & (r < 110) & (b < 110)).astype(np.uint8)
    n_lab, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    pts = centroids[1:][stats[1:, 4] >= 2].astype(int)
    if len(pts) < 50:
        pytest.skip(f"only {len(pts)} keypoint blobs detected; visualizer style may have changed")

    def sample(points):
        vals = []
        for x, y in points:
            if 0 <= x < response.shape[1] and 0 <= y < response.shape[0]:
                vals.append(response[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max())
        return np.asarray(vals)

    rng = np.random.default_rng(0)
    control = sample(np.stack([rng.uniform(0, response.shape[1], 2000),
                               rng.uniform(0, response.shape[0], 2000)], 1).astype(int))
    actual = sample(pts)

    ratio = float(np.median(actual)) / max(float(np.median(control)), 1e-12)
    assert ratio >= 10.0, (
        f"rendered keypoints score only {ratio:.1f}x a random-position control on "
        "corner response (expected >=10x). They are no longer landing on image "
        "features -- the most likely cause is scaleKeypoint falling back to the "
        "uniform ctx.scale for an aspect-distorting resize."
    )
