"""
Tiled super-resolution: tile geometry and pipelined tile execution.

Models compiled at a small fixed input (e.g. ESPCN at 17x17) can only cover a
larger image by tiling. Two things decide whether that looks right and runs fast:

**Overlap.** A CNN's output pixel needs its whole receptive field. Cut tiles with
no overlap and every pixel near a tile edge is reconstructed from the model's
internal zero-padding instead of the real neighbours, so a grid appears at the
tile period. Cutting with a ``halo`` of at least the receptive-field radius and
keeping only the valid centre removes the seams *by construction* — no blending
required. For ESPCN (conv 5x5 -> 3x3 -> 3x3) the receptive field is 9x9, so the
radius, and the correct halo, is 4.

**Pipelining.** A 17x17 ESPCN forward is ~0.1 MFLOP, so a blocking ``run()`` per
tile is almost entirely call overhead (measured 0.849 ms/tile). Issuing
``run_async()`` over a bounded in-flight window measured 0.19 ms/tile — 4.5x
more throughput, which is what makes the extra tiles from overlapping affordable.

``halo=0`` reduces this module to plain non-overlapping tiling, which is what the
runners did before it existed — useful as a regression harness.
"""

import collections
import logging
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TilePlan",
    "DEFAULT_HALO",
    "MAX_HALO",
    "max_halo_for_tile",
    "resolve_halo",
    "resolve_halo_with_source",
    "resolve_runner_halo",
    "axis_windows",
    "plan_tiles",
    "run_tiles_pipelined",
    "assemble_tiles",
]

# Receptive-field radius of ESPCN (conv 5x5 -> 3x3 -> 3x3 gives RF 9x9).
DEFAULT_HALO = 4

# Highest halo accepted. The receptive-field radius is 4, so 4 px of context is
# already everything an output pixel can use — a larger halo buys no accuracy
# while the stride (tile - 2 * halo) collapses and the tile count explodes
# (17x17 tiles over a 275x150 frame: 480 tiles at halo 4, 34,706 at halo 8).
MAX_HALO = 4

# Where an explicit halo can come from, in order of precedence.
_ENV_HALO = "DXAPP_SR_TILE_HALO"
_CONFIG_KEY = "sr_tile_halo"

# In-flight window for run_async. Measured throughput plateaus around 16-32;
# past that the queue only adds latency.
DEFAULT_INFLIGHT = 16

TilePlan = collections.namedtuple(
    "TilePlan",
    # win_y/win_x  : window origin in the padded LR plane
    # src_y/src_x  : offset of the valid region inside the window (LR px)
    # valid_h/valid_w : size of the valid region (LR px)
    "win_y win_x src_y src_x valid_h valid_w",
)


def max_halo_for_tile(tile: int) -> int:
    """Largest halo that still leaves a positive stride for ``tile``."""
    return max(0, (tile - 1) // 2)


def _coerce_halo(value, source: str) -> int:
    """Parse an explicitly given halo. Raises ValueError when unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{source}: expected an integer, got {value!r}")
    try:
        halo = int(value)
    except ValueError:
        raise ValueError(f"{source}: expected an integer, got {value!r}") from None
    if halo < 0:
        raise ValueError(f"{source}: must be >= 0, got {halo}")
    return halo


def resolve_halo_with_source(default: int = DEFAULT_HALO, cli=None, config=None,
                             tile_h: Optional[int] = None,
                             tile_w: Optional[int] = None) -> Tuple[int, str]:
    """Resolve the tile halo and report where the value came from.

    Precedence: ``cli`` > ``config["sr_tile_halo"]`` > ``DXAPP_SR_TILE_HALO`` >
    ``default``. ``cli=0`` is a real value (plain tiling), not "unset" — only
    ``None`` means unset.

    An explicitly given value (CLI or config.json) is strict: a non-integer or a
    negative number raises :class:`ValueError`, so the caller can tell the user
    their value was rejected instead of silently ignoring it. A malformed
    *environment* value keeps the historical lenient behaviour and falls back to
    ``default``.

    The result is always range-checked, whatever its source: at most
    :data:`MAX_HALO` (4 — the receptive-field radius, beyond which overlap adds no
    accuracy) and, when ``tile_h`` / ``tile_w`` are given, at most what the tile
    geometry allows (``tile - 2 * halo`` must stay positive) — an out-of-range
    value would otherwise blow up inside :func:`axis_windows`.
    """
    if cli is not None:
        halo, source = _coerce_halo(cli, "--sr-tile-halo"), "--sr-tile-halo"
    elif config and config.get(_CONFIG_KEY) is not None:
        halo = _coerce_halo(config.get(_CONFIG_KEY), f"config.json '{_CONFIG_KEY}'")
        source = "config.json"
    else:
        raw = os.environ.get(_ENV_HALO)
        halo, source = default, "default"
        if raw:
            try:
                halo, source = _coerce_halo(raw, _ENV_HALO), f"env {_ENV_HALO}"
            except ValueError:
                halo, source = default, "default"

    tiles = [t for t in (tile_h, tile_w) if t]
    limit = min([MAX_HALO] + [max_halo_for_tile(t) for t in tiles])
    if halo > limit:
        if tiles and limit < MAX_HALO:
            raise ValueError(
                f"halo {halo} is too large for a {tile_w}x{tile_h} tile "
                f"(stride would be <= 0); use 0..{limit}")
        raise ValueError(
            f"halo {halo} exceeds the maximum useful overlap; use 0..{MAX_HALO} "
            f"({MAX_HALO} is the receptive-field radius — a larger halo adds no "
            f"accuracy and multiplies the tile count)")
    return halo, source


def resolve_halo(default: int = DEFAULT_HALO, cli=None, config=None,
                 tile_h: Optional[int] = None,
                 tile_w: Optional[int] = None) -> int:
    """Halo in LR pixels. See :func:`resolve_halo_with_source` for precedence.

    Called with no arguments this keeps the historical behaviour —
    ``DXAPP_SR_TILE_HALO`` when set, else ``default`` — so existing call sites are
    unaffected. ``0`` selects plain non-overlapping tiling.
    """
    return resolve_halo_with_source(default, cli, config, tile_h, tile_w)[0]


def resolve_runner_halo(cli=None, config=None, tile_h: Optional[int] = None,
                        tile_w: Optional[int] = None,
                        verbose: bool = False) -> int:
    """Runner-facing wrapper: resolve, log the source, exit on a bad user value.

    A halo the model cannot use is a configuration mistake, not a runtime
    condition to recover from, so it ends the run with a single actionable line
    instead of a traceback from inside the tiling loop.
    """
    try:
        halo, source = resolve_halo_with_source(
            cli=cli, config=config, tile_h=tile_h, tile_w=tile_w)
    except ValueError as exc:
        logger.error(f"Invalid SR tile halo — {exc}")
        sys.exit(1)
    if verbose:
        logger.info(f"SR tile halo: {halo} px (from {source})")
    return halo


def axis_windows(orig: int, tile: int, halo: int) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Lay out windows along one axis.

    Returns ``(padded_extent, [(win, src, valid), ...])`` where ``win`` is the
    window origin, ``src`` the valid region's offset inside the window and
    ``valid`` its length. The valid regions are contiguous and together cover
    ``[0, padded_extent)``.

    A window sitting at the start of the plane, or ending exactly on its far
    edge, keeps its border pixels: there is no neighbouring tile to take that
    context from, so the model's own padding should act there — the same thing it
    sees when the whole image is processed in one pass.
    """
    stride = tile - 2 * halo
    if stride <= 0:
        raise ValueError(f"halo {halo} too large for tile {tile} (stride would be {stride})")

    steps = 0 if orig <= tile else -(-(orig - tile) // stride)  # ceil division
    padded = tile + steps * stride

    windows = []
    for i in range(steps + 1):
        win = i * stride
        src = 0 if win == 0 else halo
        end = tile if (win + tile == padded) else tile - halo
        windows.append((win, src, end - src))
    return padded, windows


def plan_tiles(orig_h: int, orig_w: int, tile_h: int, tile_w: int,
               halo: int) -> Tuple[int, int, List[TilePlan]]:
    """Plan the tiling of an ``orig_h x orig_w`` LR plane.

    Returns ``(padded_h, padded_w, plans)``. The caller must pad the LR plane to
    ``padded_h x padded_w`` before reading windows from it.
    """
    padded_h, rows = axis_windows(orig_h, tile_h, halo)
    padded_w, cols = axis_windows(orig_w, tile_w, halo)

    plans = [
        TilePlan(win_y=wy, win_x=wx, src_y=sy, src_x=sx, valid_h=vh, valid_w=vw)
        for (wy, sy, vh) in rows
        for (wx, sx, vw) in cols
    ]
    return padded_h, padded_w, plans


def run_tiles_pipelined(ie, prep: Callable[[np.ndarray], np.ndarray],
                        lr_plane: np.ndarray, plans: Sequence[TilePlan],
                        tile_h: int, tile_w: int,
                        inflight: int = DEFAULT_INFLIGHT) -> List[Optional[list]]:
    """Run every planned tile through ``ie``, pipelined via ``run_async``.

    Args:
        ie: InferenceEngine.
        prep: turns an ``[h, w, 1]`` tile into the tensor ``ie`` expects (dtype
            and layout); typically the runner's own input-preparation step.
        lr_plane: the padded LR plane, ``[padded_h, padded_w]``.
        plans: from :func:`plan_tiles`.
        inflight: how many jobs to keep queued.

    Returns outputs in tile order; an entry is ``None`` if that tile failed.
    """
    if inflight < 1:
        inflight = 1

    results: List[Optional[list]] = [None] * len(plans)
    # Holds (index, job_id, tensor). The tensor reference must outlive the job —
    # the engine may read the buffer until wait() returns.
    queue: collections.deque = collections.deque()

    def drain_one():
        idx, job_id, _tensor = queue.popleft()
        try:
            results[idx] = ie.wait(job_id)
        except Exception:
            results[idx] = None

    for i, p in enumerate(plans):
        tile = lr_plane[p.win_y:p.win_y + tile_h, p.win_x:p.win_x + tile_w]
        tensor = prep(np.ascontiguousarray(tile)[:, :, np.newaxis])
        queue.append((i, ie.run_async([tensor]), tensor))
        if len(queue) >= inflight:
            drain_one()

    while queue:
        drain_one()

    return results


def assemble_tiles(plans: Sequence[TilePlan], outputs: Sequence[Optional[list]],
                   padded_out_h: int, padded_out_w: int,
                   scale_y: int, scale_x: int) -> Tuple[np.ndarray, int]:
    """Stitch tile outputs into one SR luminance plane.

    Only each tile's valid region is written, so with a halo of at least the
    receptive-field radius the result has no tile seams. Returns
    ``(sr_y, tiles_done)``.
    """
    sr_y = np.zeros((padded_out_h, padded_out_w), dtype=np.uint8)
    tiles_done = 0

    for p, out in zip(plans, outputs):
        # ``None`` marks a tile whose wait() raised (see run_tiles_pipelined); an
        # empty list means the engine returned no output tensors. Neither carries
        # pixels to stitch, and indexing an empty list would raise, so both are
        # skipped — spelled out rather than relying on falsiness.
        if out is None or len(out) == 0:
            continue
        arr = np.squeeze(out[0])
        if arr is None or arr.ndim == 0:
            continue
        arr2d = arr[0] if arr.ndim == 3 else arr
        tile_u8 = (np.clip(arr2d, 0.0, 1.0) * 255.0).astype(np.uint8)

        sy, sx = p.src_y * scale_y, p.src_x * scale_x
        vh, vw = p.valid_h * scale_y, p.valid_w * scale_x
        src = tile_u8[sy:sy + vh, sx:sx + vw]

        dy = (p.win_y + p.src_y) * scale_y
        dx = (p.win_x + p.src_x) * scale_x
        sr_y[dy:dy + src.shape[0], dx:dx + src.shape[1]] = src
        tiles_done += 1

    # A dropped tile leaves its region black. tiles_done already reports this, but
    # only as part of an info line, so say it out loud — a partially reconstructed
    # frame should not look like a model quality problem.
    if tiles_done < len(plans):
        logger.warning(
            f"SR tiling: {len(plans) - tiles_done} of {len(plans)} tiles produced "
            f"no usable output; those regions stay black")

    return sr_y, tiles_done
